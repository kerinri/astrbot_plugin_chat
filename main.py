import os
import re
import math
import random
import asyncio
from typing import List, Dict

from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import LLMResponse
from astrbot.api.message_components import Plain, BaseMessageComponent, Image, At, Face, Reply, Node

class MessageSplitterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.pair_map = {
            '“': '”', '《': '》', '（': '）', '(': ')', 
            '[': ']', '{': '}'
        }
        self.quote_chars = {'"', "'", "`"}

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        setattr(event, "__is_llm_reply", True)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        # 1. 基础防重入与校验
        if getattr(event, "__splitter_processed", False):
            return

        result = event.get_result()
        if not result or not result.chain:
            return

        # 2. 作用范围检查
        # split_scope: 'llm_only' (默认) 或 'all'
        split_scope = self.config.get("split_scope", "llm_only")
        is_llm_reply = getattr(event, "__is_llm_reply", False)

        if split_scope == "llm_only" and not is_llm_reply:
            return

        # 3. 长度限制检查
        # max_length_no_split: 超过此长度则强制不分段，0表示不限制
        max_len_no_split = self.config.get("max_length_no_split", 0)
        total_text_len = sum(len(c.text) for c in result.chain if isinstance(c, Plain))

        if max_len_no_split > 0 and total_text_len > max_len_no_split:
            logger.info(f"[Splitter] 文本总长({total_text_len}) 超过设定阈值({max_len_no_split})，为保证完整性，取消分段与清理。")
            return

        # 标记已处理
        setattr(event, "__splitter_processed", True)

        # 4. 获取基础配置
        split_mode = self.config.get("split_mode", "regex")
        if split_mode == "simple":
            split_chars = self.config.get("split_chars", "。？！?!；;\n")
            split_pattern = f"[{re.escape(split_chars)}]+"
        else:
            split_pattern = self.config.get("split_regex", r"[。？！?!\\n…]+")

        clean_pattern = self.config.get("clean_regex", "")
        smart_mode = self.config.get("enable_smart_split", True)
        max_segs = self.config.get("max_segments", 7)
        # 兼容新旧配置键名
        forward_threshold = self.config.get("forward_threshold") or self.config.get("description", 0)
        relay_group_id = self.config.get("relay_group_id", 0)

        # 5. 获取组件策略配置
        enable_reply = self.config.get("enable_reply", True)

        # 策略选项: '跟随下段', '跟随上段', '单独', '嵌入'
        strategies = {
            'image': self.config.get("image_strategy", "单独"),
            'at': self.config.get("at_strategy", "跟随下段"),
            'face': self.config.get("face_strategy", "嵌入"),
            'default': self.config.get("other_media_strategy", "跟随下段")
        }

        # 6. 执行分段
        # 注意：此时 result.chain 中通常不包含 Reply 组件，因为框架还没加
        segments = self.split_chain_smart(result.chain, split_pattern, smart_mode, strategies, enable_reply)
        
        logger.info(f"[Splitter] 分段完成,共 {len(segments)} 段,转发阈值: {forward_threshold}")

        # 6.5 预先检测分段总数,超限时改用群合并转发
        # 重要:此处必须提前return,避免后续逐段发送逻辑执行
        if forward_threshold > 0 and len(segments) > forward_threshold:
            logger.info(f"[Splitter] ⚠️ 分段数({len(segments)}) 超过转发阈值({forward_threshold}),将使用群合并转发(不逐段发送)。")
            setattr(event, "__splitter_using_forward", True)  # 标记使用转发模式
            try:
                # 优先通过中转群：先把分段发到中转群，再转发最近消息
                if relay_group_id:
                    sent = await self._forward_via_relay_group(event, segments, clean_pattern, relay_group_id)
                    if sent:
                        result_obj = event.get_result()
                        if result_obj and result_obj.chain:
                            result_obj.chain.clear()
                        logger.info(f"[Splitter] ✓ 通过中转群({relay_group_id}) 合并转发成功，段数: {len(segments)}。")
                        return
                    logger.warning("[Splitter] 中转群合并转发失败或未获取到消息ID，将回退 chain_result。")

                logger.info("[Splitter] relay 未可用，改用 chain_result。")
                sent = self._forward_via_event_chain_result(event, segments, clean_pattern, enable_reply)
                if sent:
                    logger.info(f"[Splitter] ✓ 通过 chain_result 设置合并转发，段数: {len(segments)}。")
                    return
                logger.warning("[Splitter] chain_result 合并转发未成功，保留原始消息。")
            except Exception as e:
                logger.error(f"[Splitter] ✗ 合并转发构建失败: {e}", exc_info=True)
            return  # 关键:直接返回,不再执行后续逐段发送

        # 7. 最大分段数限制
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

        # 8. 手动注入 Reply 组件
        # 因为即将清空 result.chain，框架的自动引用逻辑会被跳过
        # 如果开启了引用，需要手动将其加到第一段的开头
        if enable_reply and segments and event.message_obj.message_id:
            # 检查第一段是否已经有 Reply (防止重复)
            has_reply = any(isinstance(c, Reply) for c in segments[0])
            if not has_reply:
                reply_comp = Reply(id=event.message_obj.message_id)
                segments[0].insert(0, reply_comp)

        # 安全检查:如果已使用转发模式,不应执行到此处
        if getattr(event, "__splitter_using_forward", False):
            logger.error(f"[Splitter] ⚠️ 逻辑错误:已使用转发模式但仍进入逐段发送,中止执行。")
            return
        
        logger.info(f"[Splitter] 将逐段发送 {len(segments)} 个分段。")

        # 9. 逐段处理与发送
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

        # 10. 清空原始链
        # 这会导致框架的 ResultDecorateStage 认为没有内容可发，从而跳过后续处理（包括自动加引用）防止重复发送
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

    def _forward_via_event_chain_result(self, event: AstrMessageEvent, segments: List[List[BaseMessageComponent]], clean_pattern: str, enable_reply: bool) -> bool:
        """通过修改 result.chain 注入 Node 列表，触发合并转发，返回是否设置成功"""
        prepared_segments: List[List[BaseMessageComponent]] = []
        for seg in segments:
            cleaned = self._clean_segment_for_forward(seg, clean_pattern)
            if cleaned:
                prepared_segments.append(cleaned)

        if not prepared_segments:
            return False

        sender_uin = getattr(event.message_obj, "self_id", None) or getattr(event.message_obj, "self_uid", None) or getattr(event.message_obj, "user_id", None) or 0
        sender_name = getattr(event.message_obj, "self_name", None) or getattr(event.message_obj, "self_nickname", None)
        if sender_name is None:
            sender = getattr(event.message_obj, "sender", None)
            if isinstance(sender, dict):
                sender_name = sender.get("nickname") or sender.get("card")
        if sender_name is None:
            sender_name = "AstrBot"

        nodes = [Node(uin=sender_uin, name=sender_name, content=seg) for seg in prepared_segments]

        if enable_reply and event.message_obj.message_id and nodes:
            first_node_content = nodes[0].content
            if not any(isinstance(c, Reply) for c in first_node_content):
                first_node_content.insert(0, Reply(id=event.message_obj.message_id))

        logger.info(f"[Splitter] 使用 event.chain_result 发送合并转发: {len(nodes)} 个Node，发送者: {sender_name}({sender_uin})")
        # 直接替换 result.chain，由框架后续阶段负责发送
        result_obj = event.get_result()
        if result_obj is None:
            return False

        result_obj.chain = nodes
        return True

    async def _forward_via_relay_group(self, event: AstrMessageEvent, segments: List[List[BaseMessageComponent]], clean_pattern: str, relay_group_id: int) -> bool:
        """将分段先发送到中转群，再把这些消息作为聊天记录合并转发到目标会话。"""
        client = getattr(event, "bot", None)
        if not client or not hasattr(client, "send_group_msg"):
            return False

        # 目标会话信息
        target_group_id = getattr(event, "group_id", None) or getattr(getattr(event, "message_obj", None), "group_id", None)
        target_user_id = (
            getattr(event, "user_id", None)
            or getattr(event, "sender_id", None)
            or getattr(getattr(event, "message_obj", None), "user_id", None)
            or getattr(getattr(event, "message_obj", None), "sender_id", None)
        )
        if not target_user_id:
            sender_info = getattr(getattr(event, "message_obj", None), "sender", None)
            if isinstance(sender_info, dict):
                target_user_id = sender_info.get("user_id") or sender_info.get("userId") or sender_info.get("id")
            else:
                target_user_id = (
                    getattr(sender_info, "user_id", None)
                    or getattr(sender_info, "userId", None)
                    or getattr(sender_info, "id", None)
                )
        can_forward_group = hasattr(client, "send_group_forward_msg")
        can_forward_private = hasattr(client, "send_private_forward_msg")

        logger.info(
            f"[Splitter] Relay目标解析 group_id={target_group_id}, user_id={target_user_id}, "
            f"can_forward_group={can_forward_group}, can_forward_private={can_forward_private}"
        )

        if target_group_id and not can_forward_group:
            return False
        if not target_group_id and not target_user_id:
            logger.warning("[Splitter] Relay: 无法获取目标用户ID，跳过中转群转发。")
            return False
        if not target_group_id and target_user_id and not can_forward_private:
            return False

        # 清洗并发送到中转群，收集 message_id
        prepared_segments: List[List[BaseMessageComponent]] = []
        for seg in segments:
            cleaned = self._clean_segment_for_forward(seg, clean_pattern)
            if cleaned:
                prepared_segments.append(cleaned)

        if not prepared_segments:
            return False

        message_ids = []
        for seg in prepared_segments:
            try:
                ob_segments = self._to_onebot_segments(seg)
                if ob_segments:
                    resp = await client.send_group_msg(group_id=int(relay_group_id), message=ob_segments)
                else:
                    # 退化为纯文本
                    text_parts = []
                    for comp in seg:
                        if isinstance(comp, Plain):
                            text_parts.append(comp.text)
                        else:
                            text_parts.append(f"[{type(comp).__name__}]")
                    content_text = "".join(text_parts).strip()
                    if not content_text:
                        continue
                    resp = await client.send_group_msg(group_id=int(relay_group_id), message=content_text)
                msg_id = None
                if isinstance(resp, dict):
                    data = resp.get("data", {}) if isinstance(resp.get("data", {}), dict) else {}
                    msg_id = (
                        resp.get("message_id")
                        or resp.get("msg_id")
                        or resp.get("id")
                        or data.get("message_id")
                        or data.get("msg_id")
                        or data.get("id")
                        or data.get("message", {}).get("message_id")
                        or data.get("message", {}).get("id")
                        or data.get("msg", {}).get("id")
                    )
                else:
                    msg_id = getattr(resp, "message_id", None) or getattr(resp, "msg_id", None) or getattr(resp, "id", None) or resp
                if msg_id:
                    message_ids.append(msg_id)
                else:
                    logger.warning(f"[Splitter] 中转群发送返回无 message_id，resp={resp}")
            except Exception as e:
                logger.error(f"[Splitter] 中转群发送失败: {e}", exc_info=True)

        if not message_ids:
            return False

        nodes = [{"type": "node", "data": {"id": mid}} for mid in message_ids]

        try:
            if target_group_id and can_forward_group:
                await client.send_group_forward_msg(group_id=int(target_group_id), messages=nodes)
                return True
            if target_user_id and can_forward_private:
                await client.send_private_forward_msg(user_id=int(target_user_id), messages=nodes)
                return True
        except Exception as e:
            logger.error(f"[Splitter] 中转群合并转发失败: {e}", exc_info=True)
            return False

        return False

    def _to_onebot_segments(self, seg: List[BaseMessageComponent]):
        """将内部组件转换为 OneBot V11 段列表，尽量保留图片/at/表情/回复等。"""
        ob: List[Dict] = []
        for comp in seg:
            if isinstance(comp, Plain):
                ob.append({"type": "text", "data": {"text": comp.text}})
            elif isinstance(comp, Image):
                file_val = getattr(comp, "file", None) or getattr(comp, "path", None)
                url_val = getattr(comp, "url", None)
                if url_val:
                    ob.append({"type": "image", "data": {"file": url_val}})
                elif file_val and os.path.exists(file_val):
                    ob.append({"type": "image", "data": {"file": file_val}})
                elif file_val:
                    logger.warning(f"[Splitter] Image file not found, fallback to placeholder: {file_val}")
                    ob.append({"type": "text", "data": {"text": "[Image]"}})
                else:
                    ob.append({"type": "text", "data": {"text": "[Image]"}})
            elif isinstance(comp, At):
                qq_val = getattr(comp, "qq", None) or getattr(comp, "id", None) or getattr(comp, "user_id", None)
                if qq_val is None:
                    ob.append({"type": "text", "data": {"text": "[At]"}})
                else:
                    ob.append({"type": "at", "data": {"qq": qq_val}})
            elif isinstance(comp, Face):
                face_id = getattr(comp, "face_id", None) or getattr(comp, "id", None) or getattr(comp, "code", None)
                if face_id is None:
                    ob.append({"type": "text", "data": {"text": "[Face]"}})
                else:
                    ob.append({"type": "face", "data": {"id": face_id}})
            elif isinstance(comp, Reply):
                reply_id = getattr(comp, "id", None) or getattr(comp, "message_id", None)
                if reply_id is None:
                    ob.append({"type": "text", "data": {"text": "[Reply]"}})
                else:
                    ob.append({"type": "reply", "data": {"id": reply_id}})
            else:
                ob.append({"type": "text", "data": {"text": f"[{type(comp).__name__}]"}})
        return ob

    async def _send_as_forward(self, event: AstrMessageEvent, segments: List[List[BaseMessageComponent]], clean_pattern: str, enable_reply: bool):
        # 先整理并清洗段落内容
        prepared_segments: List[List[BaseMessageComponent]] = []
        for seg in segments:
            cleaned = self._clean_segment_for_forward(seg, clean_pattern)
            if cleaned:
                prepared_segments.append(cleaned)

        if not prepared_segments:
            return

        # 保留引用
        if enable_reply and event.message_obj.message_id:
            first_seg = prepared_segments[0]
            if not any(isinstance(c, Reply) for c in first_seg):
                first_seg.insert(0, Reply(id=event.message_obj.message_id))

        sender_uin = getattr(event.message_obj, "self_id", None) or getattr(event.message_obj, "self_uid", None) or getattr(event.message_obj, "user_id", None) or 0
        sender_name = getattr(event.message_obj, "self_name", None) or getattr(event.message_obj, "self_nickname", None)
        if sender_name is None:
            sender = getattr(event.message_obj, "sender", None)
            if isinstance(sender, dict):
                sender_name = sender.get("nickname") or sender.get("card")
        if sender_name is None:
            sender_name = "AstrBot"

        # 构建Node列表 - 每个segment作为一个Node
        nodes = [Node(uin=sender_uin, name=sender_name, content=seg) for seg in prepared_segments]
        
        logger.info(f"[Splitter] 构建合并转发消息: {len(nodes)} 个Node节点，发送者: {sender_name}({sender_uin})")

        # 方式1: 尝试直接通过result发送（推荐）
        result = event.get_result()
        if result:
            result.chain = nodes
            logger.info(f"[Splitter] 通过result.chain发送 {len(nodes)} 个Node节点")
        else:
            # 方式2: 如果result不可用，回退到context.send_message
            mc = MessageChain()
            mc.chain = nodes
            logger.info(f"[Splitter] 通过MessageChain发送 {len(mc.chain)} 个组件")
            await self.context.send_message(event.unified_msg_origin, mc)

    def _clean_segment_for_forward(self, segment: List[BaseMessageComponent], clean_pattern: str) -> List[BaseMessageComponent]:
        cleaned: List[BaseMessageComponent] = []
        for comp in segment:
            if isinstance(comp, Plain):
                text = comp.text or ""
                if clean_pattern:
                    text = re.sub(clean_pattern, "", text)
                if not text:
                    continue
                cleaned.append(Plain(text))
            else:
                cleaned.append(comp)
        return cleaned

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
            
            # 处理引号和反引号 (支持 ` " ')
            # 如果是这些字符，它们既是开头也是结尾（对于同一种字符来说）
            if char in self.quote_chars:
                if stack and stack[-1] == char:
                    # 栈顶是相同的引号/反引号 -> 闭合
                    stack.pop()
                    current_chunk += char
                    i += 1; continue
                else:
                    # 入栈 (开启引用)
                    stack.append(char)
                    current_chunk += char
                    i += 1; continue
            
            # 处理成对的不同符号 (如 (), [], {})
            if stack:
                expected_closer = self.pair_map.get(stack[-1])
                if char == expected_closer:
                    stack.pop()
                elif is_opener:
                    stack.append(char)
                current_chunk += char
                i += 1; continue
            
            if is_opener:
                stack.append(char)
                current_chunk += char
                i += 1; continue

            # 只有在栈为空时（不在引用/括号内），才尝试匹配分段符
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
