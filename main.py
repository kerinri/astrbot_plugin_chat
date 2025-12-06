import re
import math
import random
import asyncio
from typing import List, Dict

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import Plain, BaseMessageComponent, Image, At, Face, Reply

@register("message_splitter", "YourName", "智能消息分段插件", "1.5.0")
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
        enable_reply = self.config.get("enable_reply", True)

        # 策略选项: '跟随下段', '跟随上段', '单独', '嵌入'
        strategies = {
            'image': self.config.get("image_strategy", "单独"),
            'at': self.config.get("at_strategy", "跟随下段"),
            'face': self.config.get("face_strategy", "嵌入"),
            'default': self.config.get("other_media_strategy", "跟随下段")
        }

        # 4. 执行分段
        # 注意：此时 result.chain 中通常不包含 Reply 组件，因为框架还没加
        segments = self.split_chain_smart(result.chain, split_pattern, smart_mode, strategies, enable_reply)

        # 5. 最大分段数限制
        if len(segments) > max_segs and max_segs > 0:
            logger.warning(f"[Splitter] 分段数({len(segments)}) 超过限制({max_segs})，正在合并剩余段落。")
            merged_last_segment = []
            trimmed_segments = segments[:max_segs-1]
            for seg in segments[max_segs-1:]:
                merged_last_segment.extend(seg)
            trimmed_segments.append(merged_last_segment)
            segments = trimmed_segments

        # 如果只有一段且不需要清理，直接放行 (让框架去处理 Reply)
        if len(segments) <= 1 and not clean_pattern:
            return

        # 6. 手动注入 Reply 组件
        # 因为我们即将清空 result.chain，框架的自动引用逻辑会被跳过
        # 所以如果开启了引用，我们需要手动将其加到第一段的开头
        if enable_reply and segments and event.message_obj.message_id:
            # 检查第一段是否已经有 Reply (防止重复)
            has_reply = any(isinstance(c, Reply) for c in segments[0])
            if not has_reply:
                reply_comp = Reply(id=event.message_obj.message_id)
                segments[0].insert(0, reply_comp)

        logger.info(f"[Splitter] 将发送 {len(segments)} 个分段。")

        # 7. 逐段处理与发送
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

        # 8. 清空原始链
        # 这会导致框架的 ResultDecorateStage 认为没有内容可发，从而跳过后续处理（包括自动加引用）
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

    def split_chain_smart(self, chain: List[BaseMessageComponent], pattern: str, smart_mode: bool, strategies: Dict[str, str], enable_reply: bool) -> List[List[BaseMessageComponent]]:
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
                c_type = type(component).__name__.lower()
                
                # 如果链中已经存在 Reply 组件 (可能是其他插件加的)，根据开关决定去留
                if 'reply' in c_type:
                    if enable_reply:
                        current_chain_buffer.append(component)
                    continue

                # 映射到具体的策略键
                if 'image' in c_type: strategy = strategies['image']
                elif 'at' in c_type: strategy = strategies['at']
                elif 'face' in c_type: strategy = strategies['face']
                else: strategy = strategies['default']

                if strategy == "单独":
                    if current_chain_buffer:
                        segments.append(current_chain_buffer[:])
                        current_chain_buffer.clear()
                    segments.append([component])
                    
                elif strategy == "跟随上段":
                    if current_chain_buffer:
                        current_chain_buffer.append(component)
                    elif segments:
                        segments[-1].append(component)
                    else:
                        current_chain_buffer.append(component)
                        
                else: 
                    # 跟随下段 或 嵌入
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
