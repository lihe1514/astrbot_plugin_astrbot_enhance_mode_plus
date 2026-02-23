import asyncio
import datetime
import random
import re
import traceback
import uuid
from collections import OrderedDict, defaultdict

from astrbot.api import logger, sp, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart


class Main(star.Star):
    _DEFAULT_MAX_ORIGINS = 500

    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.session_chats: dict[str, list[str]] = defaultdict(list)
        self.active_reply_stacks: dict[str, list[str]] = defaultdict(list)
        self.model_choice_histories: dict[str, list[str]] = defaultdict(list)
        self._origin_lru: OrderedDict[str, None] = OrderedDict()

    def _timeout_cfg(self) -> dict[str, float]:
        tc = self.config.get("timeouts", {})

        def _as_pos_float(val, default: float) -> float:
            try:
                x = float(val)
                return x if x > 0 else default
            except (TypeError, ValueError):
                return default

        return {
            "image_caption_sec": _as_pos_float(tc.get("image_caption_sec", 45), 45.0),
            "model_choice_sec": _as_pos_float(tc.get("model_choice_sec", 45), 45.0),
        }

    def _lru_cfg(self) -> dict[str, int]:
        lru = self.config.get("lru_cache", {})
        try:
            max_origins = int(lru.get("max_origins", self._DEFAULT_MAX_ORIGINS))
        except (TypeError, ValueError):
            max_origins = self._DEFAULT_MAX_ORIGINS
        return {
            "max_origins": max(1, max_origins),
        }

    def _evict_origin_state(self, origin: str) -> None:
        self.session_chats.pop(origin, None)
        self.active_reply_stacks.pop(origin, None)
        self.model_choice_histories.pop(origin, None)

    def _touch_origin(self, origin: str) -> None:
        if not origin:
            return
        self._origin_lru.pop(origin, None)
        self._origin_lru[origin] = None
        max_origins = self._lru_cfg()["max_origins"]
        while len(self._origin_lru) > max_origins:
            oldest, _ = self._origin_lru.popitem(last=False)
            self._evict_origin_state(oldest)

    def _react_mode_cfg(self):
        rm = self.config.get("react_mode", {})
        return {
            "enable": rm.get("enable", False),
        }

    def _group_context_cfg(self):
        gc = self.config.get("group_context", {})
        react_mode_enable = self._react_mode_cfg()["enable"]
        return {
            "enable": gc.get("enable", False) and react_mode_enable,
            "max_messages": gc.get("max_messages", 300),
            "include_sender_id": gc.get("include_sender_id", True),
            "include_role_tag": gc.get("include_role_tag", True),
            "image_caption": gc.get("image_caption", False),
            "image_caption_provider_id": gc.get("image_caption_provider_id", ""),
            "image_caption_prompt": gc.get(
                "image_caption_prompt", "Describe this image in one sentence."
            ),
        }

    def _active_reply_cfg(self):
        ar = self.config.get("active_reply", {})
        react_mode_enable = self._react_mode_cfg()["enable"]
        whitelist_raw = ar.get("whitelist", "")
        if isinstance(whitelist_raw, str):
            whitelist = [w.strip() for w in whitelist_raw.split(",") if w.strip()]
        elif isinstance(whitelist_raw, (list, tuple, set)):
            whitelist = [str(w).strip() for w in whitelist_raw if str(w).strip()]
        else:
            whitelist = []

        mode = str(ar.get("mode", "probability")).strip().lower()
        if mode not in ("probability", "model_choice"):
            mode = "probability"

        try:
            model_stack_size = int(ar.get("model_stack_size", 8))
        except (TypeError, ValueError):
            model_stack_size = 8
        model_stack_size = max(1, model_stack_size)

        try:
            model_history_messages = int(ar.get("model_history_messages", 0))
        except (TypeError, ValueError):
            model_history_messages = 0
        model_history_messages = max(0, model_history_messages)

        return {
            "enable": ar.get("enable", False) and react_mode_enable,
            "mode": mode,
            "possibility": ar.get("possibility", 0.1),
            "whitelist": whitelist,
            "model_stack_size": model_stack_size,
            "model_history_messages": model_history_messages,
            "model_choice_prompt": ar.get(
                "model_choice_prompt",
                (
                    "你当前的人格面具是：{persona_name}\n"
                    "人格设定如下：\n{persona_mask}\n\n"
                    "你正在群聊中扮演助手。以下是最近 {stack_size} 条群聊消息：\n"
                    "{messages}\n\n"
                    "额外历史上下文（最近 {history_count} 条）：\n"
                    "{history_context}\n\n"
                    "请严格站在该人格的角度判断你是否应该主动回复。"
                    "如果需要回复，只输出 REPLY；如果不需要回复，只输出 SKIP。"
                ),
            ),
        }

    def _allow_active_reply(self, event: AstrMessageEvent, ar: dict) -> bool:
        if not ar["enable"]:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if ar["whitelist"] and (
            event.unified_msg_origin not in ar["whitelist"]
            and (event.get_group_id() and event.get_group_id() not in ar["whitelist"])
        ):
            return False
        return True

    async def _resolve_persona_mask(self, event: AstrMessageEvent) -> tuple[str, str]:
        """Resolve effective persona (same priority as main agent) for model-choice judging."""
        persona_id = ""
        try:
            session_service_config = await sp.get_async(
                scope="umo",
                scope_id=event.unified_msg_origin,
                key="session_service_config",
                default={},
            )
            if isinstance(session_service_config, dict):
                persona_id = str(session_service_config.get("persona_id") or "").strip()
        except Exception as e:
            logger.debug(f"enhance-mode | 获取 session persona 失败: {e}")

        if not persona_id:
            try:
                curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if curr_cid:
                    conv = await self.context.conversation_manager.get_conversation(
                        event.unified_msg_origin,
                        curr_cid,
                    )
                    if conv and conv.persona_id:
                        persona_id = str(conv.persona_id).strip()
            except Exception as e:
                logger.debug(f"enhance-mode | 获取 conversation persona 失败: {e}")

        if not persona_id:
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            persona_id = str(
                cfg.get("provider_settings", {}).get("default_personality") or ""
            ).strip()

        if persona_id == "[%None]":
            return "none", "No persona mask."

        persona = None
        if persona_id:
            try:
                persona = next(
                    (
                        p
                        for p in self.context.persona_manager.personas_v3
                        if p.get("name") == persona_id
                    ),
                    None,
                )
            except Exception:
                persona = None

        if not persona:
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(
                    event.unified_msg_origin
                )
            except Exception:
                persona = {"name": "default", "prompt": ""}

        persona_name = str(persona.get("name") or "default")
        persona_prompt = str(persona.get("prompt") or "").strip()
        if not persona_prompt:
            persona_prompt = "You are a helpful and friendly assistant."
        return persona_name, persona_prompt

    async def _judge_model_choice(
        self,
        event: AstrMessageEvent,
        ar: dict,
        origin: str,
        messages: list[str],
        trigger_reason: str,
    ) -> bool:
        history = self.model_choice_histories[origin]
        history_context_lines = []
        if ar["model_history_messages"] > 0:
            history_context_lines = history[-ar["model_history_messages"] :]
        history_context = (
            "\n".join(history_context_lines)
            if history_context_lines
            else "(disabled or no additional history)"
        )
        logger.info(
            "enhance-mode | model_choice | 开始判定 | "
            f"origin={origin} trigger={trigger_reason} stack_size={len(messages)} "
            f"history={len(history_context_lines)}"
        )

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider or not isinstance(provider, Provider):
            logger.error("enhance-mode | 未找到可用提供商，无法执行模型选择触发")
            return False

        persona_name, persona_mask = await self._resolve_persona_mask(event)
        prompt_tmpl = str(ar["model_choice_prompt"])
        try:
            judge_prompt = prompt_tmpl.format(
                stack_size=len(messages),
                messages="\n".join(messages),
                history_count=len(history_context_lines),
                history_context=history_context,
                persona_name=persona_name,
                persona_mask=persona_mask,
            )
        except Exception:
            judge_prompt = (
                f"{prompt_tmpl}\n\n"
                f"人格面具({persona_name}):\n{persona_mask}\n\n"
                f"最近消息:\n{chr(10).join(messages)}\n\n"
                f"额外历史上下文({len(history_context_lines)}):\n{history_context}\n\n"
                "请仅输出 REPLY 或 SKIP。"
            )

        try:
            judge_resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=judge_prompt,
                    session_id=uuid.uuid4().hex,
                    persist=False,
                ),
                timeout=self._timeout_cfg()["model_choice_sec"],
            )
        except asyncio.TimeoutError:
            logger.error("enhance-mode | 模型选择触发判定超时")
            return False
        except Exception as e:
            logger.error(f"enhance-mode | 模型选择触发判定失败: {e}")
            return False

        decision_raw = (judge_resp.completion_text or "").strip().upper()
        decision = decision_raw.split()[0] if decision_raw else ""
        if decision.startswith("REPLY"):
            logger.info(
                "enhance-mode | model_choice | 判定通过(REPLY) | "
                f"origin={origin} trigger={trigger_reason} persona={persona_name}"
            )
            return True
        if decision and not decision.startswith("SKIP"):
            logger.info(
                "enhance-mode | model_choice | 判定拒绝(非标准输出按 SKIP) | "
                f"origin={origin} trigger={trigger_reason} output={decision_raw}"
            )
            return False
        logger.info(
            "enhance-mode | model_choice | 判定拒绝(SKIP) | "
            f"origin={origin} trigger={trigger_reason} persona={persona_name}"
        )
        return False

    async def _need_active_reply_model_choice(
        self, event: AstrMessageEvent, ar: dict
    ) -> bool:
        origin = event.unified_msg_origin
        self._touch_origin(origin)
        text = (event.message_str or "").strip() or "[Empty]"
        nickname = event.message_obj.sender.nickname
        sender_id = event.get_sender_id()
        stack = self.active_reply_stacks[origin]
        history = self.model_choice_histories[origin]

        stack.append(f"[{nickname}/{sender_id}]: {text}")
        history_line = (
            f"[{nickname}/{sender_id}/"
            f"{datetime.datetime.now().strftime('%H:%M:%S')}]: {text}"
        )
        history.append(history_line)
        history_limit = max(
            60,
            ar["model_stack_size"] * 6,
            ar["model_history_messages"] * 6,
        )
        if len(history) > history_limit:
            del history[:-history_limit]
        logger.info(
            "enhance-mode | model_choice | 栈填充 | "
            f"origin={origin} progress={len(stack)}/{ar['model_stack_size']} "
            f"sender={sender_id}"
        )

        if len(stack) < ar["model_stack_size"]:
            return False

        messages = stack[-ar["model_stack_size"] :]
        stack.clear()
        return await self._judge_model_choice(
            event,
            ar,
            origin,
            messages,
            trigger_reason="stack_full",
        )

    async def _get_image_caption(
        self, image_url: str, provider_id: str, prompt: str
    ) -> str:
        if not provider_id:
            provider = self.context.get_using_provider()
        else:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise Exception(f"没有找到 ID 为 {provider_id} 的提供商")
        if not isinstance(provider, Provider):
            raise Exception(f"提供商类型错误({type(provider)})，无法获取图片描述")
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                image_urls=[image_url],
                persist=False,
            ),
            timeout=self._timeout_cfg()["image_caption_sec"],
        )
        return response.completion_text

    async def _need_active_reply(self, event: AstrMessageEvent) -> bool:
        ar = self._active_reply_cfg()
        if not self._allow_active_reply(event, ar):
            return False

        if ar["mode"] == "model_choice":
            return await self._need_active_reply_model_choice(event, ar)

        return random.random() < ar["possibility"]

    @filter.on_llm_request()
    async def inject_role(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Inject user role into the existing system_reminder block."""
        if not self.config.get("role_display", True):
            return

        cfg = self.context.get_config(umo=event.unified_msg_origin)
        if not cfg.get("identifier"):
            return

        role = "admin" if event.is_admin() else "member"
        role_line = f", Role: {role}"

        # Find the existing system_reminder TextPart and inject role into it
        for part in req.extra_user_content_parts:
            if isinstance(part, TextPart) and "<system_reminder>" in part.text:
                # Insert role after the Nickname line
                if "Nickname: " in part.text and role_line not in part.text:
                    # Find the end of the Nickname value (next newline or </system_reminder>)
                    nickname_idx = part.text.index("Nickname: ")
                    # Find the end of this line
                    rest = part.text[nickname_idx:]
                    newline_idx = rest.find("\n")
                    if newline_idx != -1:
                        insert_pos = nickname_idx + newline_idx
                    else:
                        close_idx = part.text.find("</system_reminder>")
                        insert_pos = close_idx if close_idx != -1 else len(part.text)
                    part.text = (
                        part.text[:insert_pos] + role_line + part.text[insert_pos:]
                    )
                return

        # Fallback: no existing system_reminder found, add a standalone one
        reminder = f"<system_reminder>Role: {role}</system_reminder>"
        req.extra_user_content_parts.append(TextPart(text=reminder))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        """Record group messages and handle active reply."""
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        gc = self._group_context_cfg()
        if not gc["enable"] and not self._active_reply_cfg()["enable"]:
            return

        has_content = any(
            isinstance(comp, (Plain, Image, Reply))
            for comp in event.message_obj.message
        )
        if not has_content:
            return

        need_active = await self._need_active_reply(event)

        if gc["enable"]:
            try:
                await self._record_message(event, gc)
            except Exception as e:
                logger.error(f"enhance-mode | record message error: {e}")

        if need_active:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.error("enhance-mode | 未找到任何 LLM 提供商，无法主动回复")
                return
            try:
                ar = self._active_reply_cfg()
                if hasattr(event, "set_extra"):
                    event.set_extra("_enhance_active_reply_triggered", True)
                    event.set_extra("_enhance_active_reply_mode", ar["mode"])
                session_curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if not session_curr_cid:
                    logger.error(
                        "enhance-mode | 当前未处于对话状态，无法主动回复，"
                        "请使用 /switch 或 /new 创建一个会话。"
                    )
                    return

                conv = await self.context.conversation_manager.get_conversation(
                    event.unified_msg_origin,
                    session_curr_cid,
                )
                if not conv:
                    logger.error("enhance-mode | 未找到对话，无法主动回复")
                    return

                yield event.request_llm(
                    prompt=event.message_str,
                    session_id=event.session_id,
                    conversation=conv,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"enhance-mode | 主动回复失败: {e}")

    async def _record_message(self, event: AstrMessageEvent, gc: dict):
        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        nickname = event.message_obj.sender.nickname
        msg_id = event.message_obj.message_id

        if gc["include_sender_id"] and gc["include_role_tag"]:
            sender_id = event.get_sender_id()
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{sender_id}/{datetime_str}]{role_tag} #msg{msg_id}:"
        elif gc["include_sender_id"]:
            sender_id = event.get_sender_id()
            header = f"[{nickname}/{sender_id}/{datetime_str}] #msg{msg_id}:"
        elif gc["include_role_tag"]:
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{datetime_str}]{role_tag} #msg{msg_id}:"
        else:
            header = f"[{nickname}/{datetime_str}] #msg{msg_id}:"

        parts = [header]

        for comp in event.get_messages():
            if isinstance(comp, Reply):
                quote_nick = comp.sender_nickname or "Unknown"
                quote_text = (comp.message_str or "").strip() or "..."
                parts.append(f" [Quote {quote_nick}: {quote_text}]")
            elif isinstance(comp, Plain):
                parts.append(f" {comp.text}")
            elif isinstance(comp, Image):
                if gc["image_caption"]:
                    try:
                        url = comp.url if comp.url else comp.file
                        if not url:
                            raise Exception("图片 URL 为空")
                        caption = await self._get_image_caption(
                            url,
                            gc["image_caption_provider_id"],
                            gc["image_caption_prompt"],
                        )
                        parts.append(f" [Image: {caption}]")
                    except Exception as e:
                        logger.error(f"enhance-mode | 获取图片描述失败: {e}")
                        parts.append(" [Image]")
                else:
                    parts.append(" [Image]")
            elif isinstance(comp, At):
                parts.append(f" [At: {comp.name}]")

        final_message = "".join(parts)
        logger.debug(f"enhance-mode | {event.unified_msg_origin} | {final_message}")
        self._touch_origin(event.unified_msg_origin)
        self.session_chats[event.unified_msg_origin].append(final_message)
        if len(self.session_chats[event.unified_msg_origin]) > gc["max_messages"]:
            self.session_chats[event.unified_msg_origin].pop(0)

    @filter.on_llm_request()
    async def inject_group_context(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Inject recorded group chat history into system prompt."""
        gc = self._group_context_cfg()
        react_mode = self._react_mode_cfg()
        if not gc["enable"]:
            return
        if event.unified_msg_origin not in self.session_chats:
            return

        self._touch_origin(event.unified_msg_origin)
        chats_str = "\n---\n".join(self.session_chats[event.unified_msg_origin])
        bounded_chats_str = (
            f"=== CHAT_HISTORY_BEGIN ===\n{chats_str}\n=== CHAT_HISTORY_END ==="
        )

        # 构造给模型的“交互控制标签”说明。
        # 这里把 mention/quote 当成控制指令而不是容器标签，尽量减少模型输出 </mention> / </quote> 的概率。
        interaction_instructions = ""
        if self.config.get("mention_parse", True) and gc["include_sender_id"]:
            interaction_instructions += (
                "\n\n## Mention\n"
                'When you want to mention/@ a user in your reply, use a control tag: <mention id="user_id"/>.\n'
                'For example: <mention id="123456"/> Hello!\n'
                "You can mention multiple users in one message. "
                "The user_id can be found in the chat history format [nickname/user_id/time].\n"
                "Do NOT use this format for yourself.\n"
                "Important: mention tag is NOT a container tag. Do NOT output </mention>."
            )
        interaction_instructions += (
            "\n\n## Quote\n"
            'When you want to quote/reply to a specific message, place <quote id="msg_id"/> '
            "at the very beginning of your reply.\n"
            'For example: <quote id="12345"/> I agree with this!\n'
            "The msg_id can be found in the chat history after the # symbol (e.g. #msg12345).\n"
            "You can only quote ONE message per reply. The quote tag MUST be the first thing in your output.\n"
            "Only use quote when it is meaningful to reference a specific message.\n"
            "Important: quote tag is NOT a container tag. Do NOT output </quote>."
        )

        if (
            react_mode["enable"]
            and event.get_message_type() == MessageType.GROUP_MESSAGE
        ):
            is_active_triggered = event.get_extra(
                "_enhance_active_reply_triggered", False
            )
            active_mode = event.get_extra("_enhance_active_reply_mode", "")
            if is_active_triggered and active_mode == "model_choice":
                # model_choice: the model freely chooses what to reply to and how
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats_str}\n\n"
                    "You decided to actively join this conversation because some recent messages are worth replying to.\n"
                    "Choose the message(s) you want to respond to from the chat history above, "
                    "and compose a natural reply. Quote the message you choose in most cases.\n"
                    "Only output your response and do not output any other information. "
                    f"You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            else:
                # probability active reply or @-triggered: reply as a reaction
                prompt = req.prompt
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats_str}\n\n"
                    f"Now, a new message is coming: `{prompt}`. "
                    "Please react to it. Your entire output is your reply to this message. Quote the message which is coming in most cases. "
                    "Only output your response and do not output any other information. "
                    f"You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            req.contexts = []
        else:
            # 非主动回复场景：把上下文写入 system_prompt，保留原始用户问题结构。
            req.system_prompt += (
                "You are now in a chatroom. The chat history is as follows: \n"
            )
            req.system_prompt += bounded_chats_str
            req.system_prompt += interaction_instructions

    # 兼容大小写、单双引号、以及 id = "xxx" 这种带空格写法，降低模型格式漂移导致的匹配失败。
    _MENTION_RE = re.compile(
        r"""<mention\s+id\s*=\s*['"]([^'"]+)['"]\s*/?>""",
        re.IGNORECASE,
    )
    _QUOTE_RE = re.compile(
        r"""<quote\s+id\s*=\s*['"]([^'"]+)['"]\s*/?>""",
        re.IGNORECASE,
    )
    # 闭标签单独兜底：即使模型输出了错误的容器式闭标签，也要在发送前清理。
    _MENTION_CLOSE_RE = re.compile(r"</mention\s*>", re.IGNORECASE)
    _QUOTE_CLOSE_RE = re.compile(r"</quote\s*>", re.IGNORECASE)

    @staticmethod
    def _normalize_quote_id(raw_id: str | None) -> str:
        """Normalize quote id so both `12345` and `msg12345` are accepted."""
        if not raw_id:
            return ""
        quote_id = str(raw_id).strip()
        if quote_id.startswith("#"):
            quote_id = quote_id[1:]
        if quote_id.lower().startswith("msg"):
            quote_id = quote_id[3:]
        return quote_id.strip()

    @filter.on_decorating_result()
    async def parse_tags(self, event: AstrMessageEvent) -> None:
        """Parse <mention> and <quote> tags in LLM output into message components.

        For segmented reply compatibility: this hook runs at on_decorating_result,
        BEFORE text segmentation. The RespondStage will extract Reply as a header
        and only send it with the first segment, then clear it for subsequent segments.
        """
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        result = event.get_result()
        if not result or not result.chain:
            return

        # --- Phase 1: 在整条消息链中只提取“第一个 quote” ---
        # 平台语义通常只支持一次引用，因此只保留首个 quote，避免多引用产生歧义。
        quote_msg_id = None
        if any(
            isinstance(comp, Plain) and self._QUOTE_RE.search(comp.text)
            for comp in result.chain
        ):
            # Find and extract the first quote tag
            for comp in result.chain:
                if isinstance(comp, Plain):
                    m = self._QUOTE_RE.search(comp.text)
                    if m:
                        quote_msg_id = self._normalize_quote_id(m.group(1))
                        break

        # --- Phase 2: 清理 quote 并解析 mention ---
        # has_tags 不仅看开标签，也看闭标签：防止仅有 </quote> / </mention> 时提前 return 导致脏文本透传。
        parse_mention = self.config.get("mention_parse", True)
        has_tags = any(
            isinstance(comp, Plain)
            and (
                self._QUOTE_RE.search(comp.text)
                or self._QUOTE_CLOSE_RE.search(comp.text)
                or self._MENTION_CLOSE_RE.search(comp.text)
                or (parse_mention and self._MENTION_RE.search(comp.text))
            )
            for comp in result.chain
        )
        if not has_tags and not quote_msg_id:
            return

        new_chain = []
        for comp in result.chain:
            if not isinstance(comp, Plain):
                new_chain.append(comp)
                continue

            text = comp.text
            # Remove all <quote> tags from text (already extracted the ID above)
            text = self._QUOTE_RE.sub("", text)
            # 同时移除闭标签，处理模型误输出容器标签的情况。
            text = self._QUOTE_CLOSE_RE.sub("", text)

            if parse_mention and self._MENTION_RE.search(text):
                # split 后奇偶位交替为：普通文本 / mention_id / 普通文本 / mention_id ...
                parts = self._MENTION_RE.split(text)
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        # 普通文本片段里也可能夹杂孤立闭标签，继续兜底清理。
                        part = self._MENTION_CLOSE_RE.sub("", part)
                        if part.strip():
                            new_chain.append(Plain(text=part))
                    else:
                        # mention 标签转换为平台 At 组件，后续由平台完成 @ 行为。
                        new_chain.append(At(qq=part))
            else:
                # 关闭 mention_parse 时也要清理闭标签，避免原样回显给用户。
                text = self._MENTION_CLOSE_RE.sub("", text)
                if text.strip():
                    new_chain.append(Plain(text=text))

        # Insert Reply component at position 0 if a quote was found
        if quote_msg_id:
            new_chain.insert(0, Reply(id=quote_msg_id))

        result.chain = new_chain

    @filter.on_llm_response()
    async def record_bot_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """Record bot response to group chat history."""
        gc = self._group_context_cfg()
        if not gc["enable"]:
            return
        if event.unified_msg_origin not in self.session_chats:
            return
        if not resp.completion_text:
            return

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        # 写入会话历史前做一次清洗，避免控制标签污染后续 chat history 注入内容。
        text = self._MENTION_RE.sub(r"[At: \1]", resp.completion_text)
        text = self._MENTION_CLOSE_RE.sub("", text)
        text = self._QUOTE_RE.sub("", text)
        text = self._QUOTE_CLOSE_RE.sub("", text)
        text = text.strip()
        final_message = f"[You/{datetime_str}]: {text}"
        logger.debug(
            f"enhance-mode | recorded AI response: "
            f"{event.unified_msg_origin} | {final_message}"
        )
        self._touch_origin(event.unified_msg_origin)
        self.session_chats[event.unified_msg_origin].append(final_message)
        if len(self.session_chats[event.unified_msg_origin]) > gc["max_messages"]:
            self.session_chats[event.unified_msg_origin].pop(0)

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        """Clean up session chats when conversation is cleared."""
        clean_session = event.get_extra("_clean_ltm_session", False)
        if not clean_session:
            return
        if event.unified_msg_origin in self.session_chats:
            del self.session_chats[event.unified_msg_origin]
        if event.unified_msg_origin in self.active_reply_stacks:
            del self.active_reply_stacks[event.unified_msg_origin]
        if event.unified_msg_origin in self.model_choice_histories:
            del self.model_choice_histories[event.unified_msg_origin]
        self._origin_lru.pop(event.unified_msg_origin, None)
