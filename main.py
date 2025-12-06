import re
import math
import random
import asyncio
from typing import List, Dict

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import Plain, BaseMessageComponent, Image, At, Face

@register("message_splitter", "YourName", "智能消息分段插件", "1.3.0")
class MessageSplitterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.pair_map = {
            '“': '”', '《': '》', '（': '）', '(': ')', 
            '[': ']', '{': '}', '"': '"', "'": "'"
        }

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__is_llm_reply", True)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 1. 校验逻辑
        if not getattr(event, "__is_llm_reply", False):
            return
        if getattr(event, "__splitter_processed", False):
            return
        setattr(event, "__splitter_processed", True)

        result = event.get_result()
        if not result or not result.chain:
            return

        # 2. 获取基础配置
        split_mode = self.config.get("split_mode", "regex")
        if split_mode == "simple":
            split_chars = self.config.get("split_chars", "。？！?!；;\n")
            split_pattern = f"[{re.escape(split_chars)}]+"
        else:
            split_pattern = self.config.get("split_regex", r"[。？！?!\\n…]+")

        clean_pattern = self.config.get("clean_regex", "")
        smart_mode = self.config.get("enable_smart_split", True)
        max_segs = self.config.get("max_segments", 7)

        # 3. 获取组件策略配置
        # 策略选项: '跟随下段', '跟随上段', '单独', '嵌入'
        strategies = {
            'image': self.config.get("image_strategy", "单独"),
            'at': self.config.get("at_strategy", "跟随下段"),
            'face': self.config.get("face_strategy", "嵌入"),
            'reply': self.config.get("reply_strategy", "单独"),
            'default': self.config.get("other_media_strategy", "跟随下段")
        }

        # 4. 执行分段
        segments = self.split_chain_smart(result.chain, split_pattern, smart_mode, strategies)

        # 5. 最大分段数限制
        if len(segments) > max_segs and max_segs > 0:
            logger.warning(f"[Splitter] 分段数({len(segments)}) 超过限制({max_segs})，正在合并剩余段落。")
            merged_last_segment = []
            trimmed_segments = segments[:max_segs-1]
            for seg in segments[max_segs-1:]:
                merged_last_segment.extend(seg)
            trimmed_segments.append(merged_last_segment)
            segments = trimmed_segments

        # 如果只有一段且不需要清理，直接放行
        if len(segments) <= 1 and not clean_pattern:
            return

        logger.info(f"[Splitter] 将发送 {len(segments)} 个分段。")

        # 6. 逐段处理与发送
        for i, segment_chain in enumerate(segments):
            if not segment_chain:
                continue

            # 应用清理正则
            if clean_pattern:
                for comp in segment_chain:
                    if isinstance(comp, Plain) and comp.text:
                        comp.text = re.sub(clean_pattern, "", comp.text)

            # 预览与日志
            preview_text = self._get_chain_preview(segment_chain)
            text_content = "".join([c.text for c in segment_chain if isinstance(c, Plain)])
            
            # 空内容检查
            is_empty_text = not text_content
            has_other_components = any(not isinstance(c, Plain) for c in segment_chain)
            if is_empty_text and not has_other_components:
                continue

            logger.info(f"[Splitter] 发送第 {i+1}/{len(segments)} 段: {preview_text}")

            try:
                mc = MessageChain()
                mc.chain = segment_chain
                await self.context.send_message(event.unified_msg_origin, mc)

                # 延迟逻辑
                if i < len(segments) - 1:
                    wait_time = self.calculate_delay(text_content)
                    await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"[Splitter] 发送分段失败: {e}")

        # 7. 清空原始链
        result.chain.clear()

    def _get_chain_preview(self, chain: List[BaseMessageComponent]) -> str:
        parts = []
        for comp in chain:
            if isinstance(comp, Plain):
                t = comp.text.replace('\n', '\\n')
                parts.append(f"\"{t[:10]}...\"" if len(t) > 10 else f"\"{t}\"")
            else:
                parts.append(f"[{type(comp).__name__}]")
        return " ".join(parts)

    def calculate_delay(self, text: str) -> float:
        strategy = self.config.get("delay_strategy", "linear")
        
        if strategy == "random":
            mn = self.config.get("random_min", 1.0)
            mx = self.config.get("random_max", 3.0)
            return random.uniform(mn, mx)
            
        elif strategy == "log":
            base = self.config.get("log_base", 0.5)
            factor = self.config.get("log_factor", 0.8)
            return min(base + factor * math.log(len(text) + 1), 5.0)
            
        elif strategy == "linear":
            base = self.config.get("linear_base", 0.5)
            factor = self.config.get("linear_factor", 0.1)
            return base + (len(text) * factor)
            
        else: # fixed
            return self.config.get("fixed_delay", 1.5)

    def split_chain_smart(self, chain: List[BaseMessageComponent], pattern: str, smart_mode: bool, strategies: Dict[str, str]) -> List[List[BaseMessageComponent]]:
        segments = []
        current_chain_buffer = []

        for component in chain:
            # --- 文本组件处理 ---
            if isinstance(component, Plain):
                text = component.text
                if not text: continue
                
                if not smart_mode:
                    self._process_text_simple(text, pattern, segments, current_chain_buffer)
                else:
                    self._process_text_smart(text, pattern, segments, current_chain_buffer)
            
            # --- 富媒体组件处理 ---
            else:
                # 获取组件类型名称 (转小写匹配配置)
                c_type = type(component).__name__.lower()
                
                # 映射到具体的策略键
                if 'image' in c_type: strategy = strategies['image']
                elif 'at' in c_type: strategy = strategies['at']
                elif 'face' in c_type: strategy = strategies['face']
                elif 'reply' in c_type: strategy = strategies['reply']
                else: strategy = strategies['default']

                if strategy == "单独":
                    # 策略：单独成段
                    # 1. 提交当前缓冲区
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                    # 2. 提交组件本身为一段
                    segments.append([component])
                    
                elif strategy == "跟随上段":
                    # 策略：跟随上文
                    if current_chain_buffer:
                        # 如果缓冲区有内容，直接追加
                        current_chain_buffer.append(component)
                    elif segments:
                        # 如果缓冲区为空，但有前一段，追加到前一段末尾
                        segments[-1].append(component)
                    else:
                        # 如果既无缓冲也无前段（消息开头），只能放入缓冲
                        current_chain_buffer.append(component)
                        
                else: 
                    # 策略：跟随下段 (跟随下文) 或 嵌入 (嵌入)
                    # 逻辑上都是放入当前缓冲区，等待后续内容或作为新段落开头
                    current_chain_buffer.append(component)

        # 处理剩余的 buffer
        if current_chain_buffer:
            segments.append(current_chain_buffer)

        return [seg for seg in segments if seg]

    def _process_text_simple(self, text: str, pattern: str, segments: list, buffer: list):
        parts = re.split(f"({pattern})", text)
        temp_text = ""
        for part in parts:
            if not part: continue
            if re.fullmatch(pattern, part):
                temp_text += part
                buffer.append(Plain(temp_text))
                segments.append(buffer[:])
                buffer.clear()
                temp_text = ""
            else:
                if temp_text: buffer.append(Plain(temp_text))
                temp_text = part
        if temp_text: buffer.append(Plain(temp_text))

    def _process_text_smart(self, text: str, pattern: str, segments: list, buffer: list):
        stack = []
        compiled_pattern = re.compile(pattern)
        i = 0
        n = len(text)
        current_chunk = ""

        while i < n:
            char = text[i]
            is_opener = char in self.pair_map
            
            if char in ['"', "'"]:
                if stack and stack[-1] == char:
                    stack.pop()
                    current_chunk += char
                    i += 1; continue
                else:
                    stack.append(char)
                    current_chunk += char
                    i += 1; continue
            
            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer: stack.pop()
                elif is_opener: stack.append(char)
                current_chunk += char
                i += 1; continue
            
            if is_opener:
                stack.append(char)
                current_chunk += char
                i += 1; continue

            match = compiled_pattern.match(text, pos=i)
            if match:
                delimiter = match.group()
                current_chunk += delimiter
                buffer.append(Plain(current_chunk))
                segments.append(buffer[:])
                buffer.clear()
                current_chunk = ""
                i += len(delimiter)
            else:
                current_chunk += char
                i += 1

        if current_chunk:
            buffer.append(Plain(current_chunk))
