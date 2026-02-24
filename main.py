import asyncio
import datetime
import json
import random
import time
import traceback
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from astrbot.api import llm_tool, logger, sp, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .ban_control import BanStore, parse_duration_seconds
from .memory_rag_store import MemoryRAGStore
from .plugin_config import PluginConfig, parse_plugin_config
from .runtime_state import RuntimeState
from .tag_utils import (
    bounded_chat_history_text,
    build_interaction_instructions,
    chain_has_refuse_tag,
    clean_response_text_for_history,
    has_refuse_tag,
    transform_result_chain,
)
from .webui import RAGWebUIServer


class Main(star.Star):
    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.runtime = RuntimeState()
        plugin_data_dir = (
            Path(get_astrbot_data_path())
            / "plugin_data"
            / "astrbot_plugin_astrbot_enhance_mode"
        )
        self.ban_store = BanStore(plugin_data_dir / "ban_list.db")
        self.memory_rag_store: MemoryRAGStore | None = None
        self.rag_webui_server: RAGWebUIServer | None = None
        try:
            self.memory_rag_store = MemoryRAGStore(plugin_data_dir / "memory_rag.db")
        except Exception as e:
            logger.error(f"enhance-mode | 初始化记忆 RAG 存储失败: {e}", exc_info=True)

    def _cfg(self) -> PluginConfig:
        return parse_plugin_config(self.config)

    def _touch_origin(self, origin: str, cfg: PluginConfig) -> None:
        self.runtime.touch_origin(origin, cfg.global_settings.lru_cache.max_origins)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(0, int(seconds))
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)

    @staticmethod
    def _ban_scope_id(event: AstrMessageEvent) -> str:
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return ""
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return ""
        platform_id = (
            str(event.get_platform_id() or "").strip()
            or str(event.get_platform_name() or "").strip()
        )
        return f"{platform_id}:{group_id}" if platform_id else group_id

    def _get_admin_sid_set(self) -> set[str]:
        base_cfg = self.context.get_config()
        if not isinstance(base_cfg, dict):
            return set()
        admins = base_cfg.get("admins_id", [])
        if not isinstance(admins, list):
            return set()
        return {str(sid).strip() for sid in admins if str(sid).strip()}

    @staticmethod
    def _parse_role_ids(raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []

        parsed_ids: list[str] = []
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    parsed_ids = [str(item).strip() for item in payload]
            except json.JSONDecodeError:
                parsed_ids = []

        if not parsed_ids:
            normalized = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
            parsed_ids = [token.strip() for token in normalized.split(",")]

        deduped: list[str] = []
        seen: set[str] = set()
        for role_id in parsed_ids:
            if not role_id or role_id in seen:
                continue
            seen.add(role_id)
            deduped.append(role_id)
        return deduped

    @staticmethod
    def _parse_optional_timestamp(raw: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None

        try:
            numeric = float(text)
            # Heuristic: treat millisecond timestamps as ms.
            if numeric > 1e12:
                numeric /= 1000.0
            return numeric
        except (TypeError, ValueError):
            pass

        normalized = text.replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.datetime.strptime(normalized, fmt)
                return dt.timestamp()
            except ValueError:
                continue

        try:
            dt = datetime.datetime.fromisoformat(normalized)
            return dt.timestamp()
        except ValueError:
            return None

    @staticmethod
    def _parse_extra_metadata(raw: str) -> dict:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            return {"raw_metadata": payload}
        except json.JSONDecodeError:
            return {"raw_metadata": text}

    @staticmethod
    def _normalize_sort_order(raw: str) -> str:
        return "asc" if str(raw or "").strip().lower() == "asc" else "desc"

    @staticmethod
    def _normalize_sort_by(raw: str) -> str:
        return "time" if str(raw or "").strip().lower() == "time" else "relevance"

    def _resolve_memory_scope(
        self,
        event: AstrMessageEvent,
        group_scope: str,
        group_id: str,
        platform_id: str,
    ) -> tuple[str, str, str]:
        explicit_scope = str(group_scope or "").strip()
        explicit_group_id = str(group_id or "").strip()
        explicit_platform_id = str(platform_id or "").strip()

        event_group_id = (
            str(event.get_group_id() or "").strip()
            if event.get_message_type() == MessageType.GROUP_MESSAGE
            else ""
        )
        event_platform_id = str(
            event.get_platform_id() or event.get_platform_name() or ""
        ).strip()

        final_group_id = explicit_group_id or event_group_id
        final_platform_id = explicit_platform_id or event_platform_id

        if explicit_scope:
            final_scope = explicit_scope
        elif final_group_id:
            final_scope = (
                f"{final_platform_id}:{final_group_id}"
                if final_platform_id
                else final_group_id
            )
        else:
            final_scope = ""

        return final_scope, final_group_id, final_platform_id

    def _resolve_embedding_provider(
        self, cfg: PluginConfig
    ) -> EmbeddingProvider | None:
        provider_id = str(cfg.memory_rag.embedding_provider_id or "").strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider and isinstance(provider, EmbeddingProvider):
                return provider
            logger.warning(
                f"enhance-mode | 配置的 embedding_provider_id 无效或类型不匹配: {provider_id}"
            )

        all_embedding_providers = self.context.get_all_embedding_providers()
        if all_embedding_providers:
            return all_embedding_providers[0]
        return None

    def _check_memory_rag_ready(self) -> tuple[bool, str]:
        cfg = self._cfg()
        if not cfg.memory_rag.enable:
            return False, "Memory RAG is disabled in enhance mode config."
        if not self.memory_rag_store:
            return False, "Memory RAG store is not initialized."
        return True, ""

    def _memory_rag_webui_url(self, cfg: PluginConfig) -> str:
        host = str(cfg.memory_rag_webui.host or "127.0.0.1").strip() or "127.0.0.1"
        port = int(cfg.memory_rag_webui.port)
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    async def _start_memory_rag_webui(self) -> None:
        cfg = self._cfg()
        webui_cfg = cfg.memory_rag_webui
        if not webui_cfg.enable:
            return
        if not self.memory_rag_store:
            logger.warning(
                "enhance-mode | memory_rag_webui enabled but memory_rag_store is unavailable"
            )
            return
        if self.rag_webui_server is not None:
            return

        try:
            self.rag_webui_server = RAGWebUIServer(
                store=self.memory_rag_store,
                config={
                    "host": webui_cfg.host,
                    "port": webui_cfg.port,
                    "access_password": webui_cfg.access_password,
                    "session_timeout": webui_cfg.session_timeout,
                },
                plugin_version="0.1.0",
            )
            await self.rag_webui_server.start()
            logger.info(
                "enhance-mode | Memory RAG WebUI started at %s",
                self._memory_rag_webui_url(cfg),
            )
            if self.rag_webui_server.password_generated:
                logger.info(
                    "enhance-mode | Memory RAG WebUI generated password: %s",
                    self.rag_webui_server.access_password,
                )
        except Exception as e:
            logger.error(
                f"enhance-mode | 启动 Memory RAG WebUI 失败: {e}", exc_info=True
            )
            self.rag_webui_server = None

    async def _stop_memory_rag_webui(self) -> None:
        if self.rag_webui_server is None:
            return
        try:
            await self.rag_webui_server.stop()
        except Exception as e:
            logger.warning(
                f"enhance-mode | 停止 Memory RAG WebUI 失败: {e}", exc_info=True
            )
        finally:
            self.rag_webui_server = None

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        await self._start_memory_rag_webui()

    def _allow_active_reply(self, event: AstrMessageEvent, cfg: PluginConfig) -> bool:
        ar = cfg.active_reply
        if not cfg.active_reply_enabled:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if ar.whitelist and (
            event.unified_msg_origin not in ar.whitelist
            and (event.get_group_id() and event.get_group_id() not in ar.whitelist)
        ):
            return False
        return True

    async def _resolve_persona_mask(self, event: AstrMessageEvent) -> tuple[str, str]:
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
        cfg: PluginConfig,
        origin: str,
        messages: list[str],
        trigger_reason: str,
    ) -> bool:
        ar = cfg.active_reply
        history = self.runtime.model_choice_histories[origin]
        history_context_lines = []
        if ar.model_history_messages > 0:
            history_context_lines = history[-ar.model_history_messages :]
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
        prompt_tmpl = ar.model_choice_prompt
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
                timeout=cfg.global_settings.timeouts.model_choice_sec,
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
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        origin = event.unified_msg_origin
        self._touch_origin(origin, cfg)

        ar = cfg.active_reply
        text = (event.message_str or "").strip() or "[Empty]"
        nickname = event.message_obj.sender.nickname
        sender_id = event.get_sender_id()
        stack = self.runtime.active_reply_stacks[origin]
        history = self.runtime.model_choice_histories[origin]

        stack.append(f"[{nickname}/{sender_id}]: {text}")
        history_line = (
            f"[{nickname}/{sender_id}/"
            f"{datetime.datetime.now().strftime('%H:%M:%S')}]: {text}"
        )
        history.append(history_line)

        history_limit = max(
            60,
            ar.model_stack_size * 6,
            ar.model_history_messages * 6,
        )
        if len(history) > history_limit:
            del history[:-history_limit]

        logger.info(
            "enhance-mode | model_choice | 栈填充 | "
            f"origin={origin} progress={len(stack)}/{ar.model_stack_size} "
            f"sender={sender_id}"
        )

        if len(stack) < ar.model_stack_size:
            return False

        messages = stack[-ar.model_stack_size :]
        stack.clear()
        return await self._judge_model_choice(
            event,
            cfg,
            origin,
            messages,
            trigger_reason="stack_full",
        )

    async def _get_image_caption(
        self,
        image_url: str,
        provider_id: str,
        prompt: str,
        timeout_sec: float,
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
            timeout=timeout_sec,
        )
        return response.completion_text

    async def _need_active_reply(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        if not self._allow_active_reply(event, cfg):
            return False

        ar = cfg.active_reply
        if ar.mode == "model_choice":
            return await self._need_active_reply_model_choice(event, cfg)
        return random.random() < ar.possibility

    @filter.on_llm_request()
    async def inject_role(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        cfg = self._cfg()
        if not cfg.group_features.role_display:
            return

        base_cfg = self.context.get_config(umo=event.unified_msg_origin)
        if not base_cfg.get("identifier"):
            return

        role = "admin" if event.is_admin() else "member"
        role_line = f", Role: {role}"

        for part in req.extra_user_content_parts:
            if isinstance(part, TextPart) and "<system_reminder>" in part.text:
                if "Nickname: " in part.text and role_line not in part.text:
                    nickname_idx = part.text.index("Nickname: ")
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

        reminder = f"<system_reminder>Role: {role}</system_reminder>"
        req.extra_user_content_parts.append(TextPart(text=reminder))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=9999)
    async def guard_banned_user(self, event: AstrMessageEvent) -> None:
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return

        released_count = self.ban_store.cleanup_expired(scope_id=scope_id)
        if released_count > 0:
            logger.info(
                "enhance-mode | 自动解封过期用户数量 | "
                f"count={released_count} scope={scope_id}"
            )

        sender_id = str(event.get_sender_id() or "").strip()
        if not sender_id:
            return
        if (
            not cfg.group_features.ban_allow_admin
            and sender_id in self._get_admin_sid_set()
        ):
            active_ban = self.ban_store.get_active_ban(
                scope_id=scope_id, user_id=sender_id
            )
            if active_ban:
                logger.info(
                    "enhance-mode | 管理员保护生效，命中封禁但按 bypass 放行(不修改数据库) | "
                    f"user_id={sender_id} remaining={self._format_duration(active_ban.remaining_seconds)} "
                    f"scope={scope_id} origin={event.unified_msg_origin}"
                )
            return

        active_ban = self.ban_store.get_active_ban(scope_id=scope_id, user_id=sender_id)
        if not active_ban:
            return

        logger.info(
            "enhance-mode | 命中封禁名单，已拦截消息 | "
            f"user_id={sender_id} remaining={self._format_duration(active_ban.remaining_seconds)} "
            f"scope={scope_id} origin={event.unified_msg_origin}"
        )
        event.stop_event()

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        cfg = self._cfg()
        if not cfg.group_history_enabled and not cfg.active_reply_enabled:
            return

        has_content = any(
            isinstance(comp, (Plain, Image, Reply))
            for comp in event.message_obj.message
        )
        if not has_content:
            return

        need_active = await self._need_active_reply(event, cfg)

        if cfg.group_history_enabled:
            try:
                await self._record_message(event, cfg)
            except Exception as e:
                logger.error(f"enhance-mode | record message error: {e}")

        if need_active:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.error("enhance-mode | 未找到任何 LLM 提供商，无法主动回复")
                return
            try:
                if hasattr(event, "set_extra"):
                    event.set_extra("_enhance_active_reply_triggered", True)
                    event.set_extra("_enhance_active_reply_mode", cfg.active_reply.mode)

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

    async def _record_message(self, event: AstrMessageEvent, cfg: PluginConfig) -> None:
        history_cfg = cfg.group_history

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        nickname = event.message_obj.sender.nickname
        msg_id = event.message_obj.message_id

        if history_cfg.include_sender_id and history_cfg.include_role_tag:
            sender_id = event.get_sender_id()
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{sender_id}/{datetime_str}]{role_tag} #msg{msg_id}:"
        elif history_cfg.include_sender_id:
            sender_id = event.get_sender_id()
            header = f"[{nickname}/{sender_id}/{datetime_str}] #msg{msg_id}:"
        elif history_cfg.include_role_tag:
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
                if history_cfg.image_caption:
                    try:
                        url = comp.url if comp.url else comp.file
                        if not url:
                            raise Exception("图片 URL 为空")
                        caption = await self._get_image_caption(
                            url,
                            history_cfg.image_caption_provider_id,
                            history_cfg.image_caption_prompt,
                            cfg.global_settings.timeouts.image_caption_sec,
                        )
                        parts.append(f" [Image: {caption}]")
                    except Exception as e:
                        logger.exception("enhance-mode | 获取图片描述失败: %s", e)
                        parts.append(" [Image]")
                else:
                    parts.append(" [Image]")
            elif isinstance(comp, At):
                parts.append(f" [At: {comp.name}]")

        final_message = "".join(parts)
        logger.debug(f"enhance-mode | {event.unified_msg_origin} | {final_message}")

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > history_cfg.max_messages:
            chats.pop(0)

    @filter.on_llm_request()
    async def inject_group_context(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return

        self._touch_origin(event.unified_msg_origin, cfg)
        bounded_chats = bounded_chat_history_text(
            self.runtime.session_chats[event.unified_msg_origin]
        )
        interaction_instructions = build_interaction_instructions(
            cfg.group_features.mention_parse,
            cfg.group_history.include_sender_id,
        )

        if (
            cfg.group_features.react_mode_enable
            and event.get_message_type() == MessageType.GROUP_MESSAGE
        ):
            is_active_triggered = event.get_extra(
                "_enhance_active_reply_triggered", False
            )
            active_mode = event.get_extra("_enhance_active_reply_mode", "")
            if is_active_triggered and active_mode == "model_choice":
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats}\n\n"
                    "You decided to actively join this conversation because some recent messages are worth replying to.\n"
                    "Choose the message(s) you want to respond to from the chat history above, "
                    "and compose a natural reply. Quote the message you choose in most cases.\n"
                    "Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            else:
                prompt = req.prompt
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats}\n\n"
                    f"Now, a new message is coming: `{prompt}`. "
                    "Please react to it. Your entire output is your reply to this message. "
                    "Quote the message which is coming in most cases. "
                    "Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            req.contexts = []
        else:
            req.system_prompt += (
                "You are now in a chatroom. The chat history is as follows: \n"
            )
            req.system_prompt += bounded_chats
            req.system_prompt += interaction_instructions

    @filter.on_decorating_result()
    async def parse_tags(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if not result or not result.chain:
            return

        # 全局拦截：模型输出 <refuse/> 时直接清空结果链，阻止后续发送到平台。
        if chain_has_refuse_tag(result.chain):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，已取消发送 | "
                f"origin={event.unified_msg_origin}"
            )
            if hasattr(event, "set_extra"):
                event.set_extra("_enhance_refused_reply", True)
            result.chain = []
            return

        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        transformed = transform_result_chain(
            result.chain,
            parse_mention=self._cfg().group_features.mention_parse,
        )
        if transformed is None:
            return

        result.chain = transformed

    @filter.on_llm_response()
    async def record_bot_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return
        if not resp.completion_text:
            return

        if has_refuse_tag(resp.completion_text):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，跳过机器人回复历史记录 | "
                f"origin={event.unified_msg_origin}"
            )
            return

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        text = clean_response_text_for_history(resp.completion_text)
        if not text:
            return
        final_message = f"[You/{datetime_str}]: {text}"

        logger.debug(
            f"enhance-mode | recorded AI response: "
            f"{event.unified_msg_origin} | {final_message}"
        )

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > cfg.group_history.max_messages:
            chats.pop(0)

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        clean_session = event.get_extra("_clean_ltm_session", False)
        if not clean_session:
            return
        self.runtime.cleanup_origin(event.unified_msg_origin)

    @llm_tool(name="enhance_get_ban_list_status")
    async def get_ban_list_status(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        max_results: int = 20,
    ) -> str:
        """Get ban-list status maintained by enhance mode.

        Args:
            user_id(string): Optional. Target user ID to query. Empty means list active bans.
            max_results(number): Optional. Maximum number of active bans to return when user_id is empty.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Ban list is group-scoped. "
                "Please call this tool in current group chat context."
            )

        released_count = self.ban_store.cleanup_expired(scope_id=scope_id)
        target_user_id = str(user_id or "").strip().lstrip("@")
        if (
            target_user_id
            and not cfg.group_features.ban_allow_admin
            and target_user_id in self._get_admin_sid_set()
        ):
            return (
                f"User `{target_user_id}` is AstrBot admin and protected from ban "
                f"(scope={scope_id}, auto released this round: {released_count})."
            )

        if target_user_id:
            active_ban = self.ban_store.get_active_ban(
                scope_id=scope_id, user_id=target_user_id
            )
            if not active_ban:
                return (
                    f"User `{target_user_id}` is not banned now. "
                    f"(scope={scope_id}, auto released this round: {released_count})"
                )

            expires_at_text = datetime.datetime.fromtimestamp(
                active_ban.expires_at
            ).strftime("%Y-%m-%d %H:%M:%S")
            return (
                f"User `{target_user_id}` is currently banned.\n"
                f"- Remaining: {self._format_duration(active_ban.remaining_seconds)}\n"
                f"- Expires at: {expires_at_text}\n"
                f"- Scope: {scope_id}\n"
                f"- Auto released this round: {released_count}"
            )

        try:
            parsed_max_results = int(max_results)
        except (TypeError, ValueError):
            parsed_max_results = 20
        limit = max(1, min(parsed_max_results, 200))
        records = self.ban_store.list_active_bans(scope_id=scope_id, limit=limit)
        if not records:
            return (
                f"No active bans in scope `{scope_id}`. "
                f"(auto released this round: {released_count})"
            )

        lines = [
            f"Active bans in scope `{scope_id}` ({len(records)} shown, limit={limit}). "
            f"Auto released this round: {released_count}",
        ]
        for idx, record in enumerate(records, start=1):
            expire_text = datetime.datetime.fromtimestamp(record.expires_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            lines.append(
                f"{idx}. user_id={record.user_id}, remaining={self._format_duration(record.remaining_seconds)}, "
                f"expires_at={expire_text}"
            )
        return "\n".join(lines)

    @llm_tool(name="enhance_ban_user")
    async def ban_user(
        self,
        event: AstrMessageEvent,
        user_id: str,
        duration: str = "10m",
    ) -> str:
        """Temporarily ban a user in current group scope.

        Args:
            user_id(string): Required. Target user ID to ban.
            duration(string): Optional. Ban duration like 60s, 10m, 2h, 1d. Default is 10m.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Ban action is group-scoped. "
                "Please call this tool in current group chat context."
            )

        target_user_id = str(user_id or "").strip().lstrip("@")
        if not target_user_id:
            return "Invalid `user_id`: empty."

        duration_seconds = parse_duration_seconds(duration)
        if duration_seconds is None:
            return "Invalid `duration`. Use formats like `60s`, `10m`, `2h`, `1d`."

        if (
            not cfg.group_features.ban_allow_admin
            and target_user_id in self._get_admin_sid_set()
        ):
            return (
                f"User `{target_user_id}` is AstrBot admin and protected from ban "
                f"(scope={scope_id})."
            )

        final_duration = max(
            1,
            min(duration_seconds, cfg.group_features.ban_max_duration_sec),
        )
        expires_at = self.ban_store.ban_user(
            scope_id=scope_id,
            user_id=target_user_id,
            duration_seconds=final_duration,
            source_origin=event.unified_msg_origin,
        )
        expire_time = datetime.datetime.fromtimestamp(expires_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return (
            f"Banned user `{target_user_id}` in scope `{scope_id}`.\n"
            f"- Duration: {self._format_duration(final_duration)}\n"
            f"- Expires at: {expire_time}"
        )

    @llm_tool(name="enhance_unban_user")
    async def unban_user(self, event: AstrMessageEvent, user_id: str) -> str:
        """Unban a user in current group scope.

        Args:
            user_id(string): Required. Target user ID to unban.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Unban action is group-scoped. "
                "Please call this tool in current group chat context."
            )

        target_user_id = str(user_id or "").strip().lstrip("@")
        if not target_user_id:
            return "Invalid `user_id`: empty."

        removed = self.ban_store.unban_user(scope_id=scope_id, user_id=target_user_id)
        if not removed:
            return f"User `{target_user_id}` is not banned in scope `{scope_id}`."
        return f"Unbanned user `{target_user_id}` in scope `{scope_id}`."

    @llm_tool(name="enhance_memory_rag_write")
    async def memory_rag_write(
        self,
        event: AstrMessageEvent,
        content: str,
        related_role_ids: str,
        memory_time: str = "",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        extra_metadata_json: str = "{}",
    ) -> str:
        """Write a memory record into enhance-mode memory RAG store.

        Args:
            content(string): Required. Memory content text.
            related_role_ids(string): Required. Role IDs in JSON array or comma-separated string.
            memory_time(string): Optional. Memory time as unix timestamp or ISO datetime.
            group_scope(string): Optional. Full group scope, for example `qq:123456`.
            group_id(string): Optional. Group ID, combined with platform_id when group_scope is empty.
            platform_id(string): Optional. Platform ID used with group_id.
            extra_metadata_json(string): Optional. JSON object string for extra metadata.
        """
        ready, reason = self._check_memory_rag_ready()
        if not ready:
            return reason
        if self.memory_rag_store is None:
            return "Memory RAG store is not initialized."

        clean_content = str(content or "").strip()
        if not clean_content:
            return "Invalid `content`: empty."

        role_ids = self._parse_role_ids(related_role_ids)
        if not role_ids:
            return "Invalid `related_role_ids`: at least one role ID is required."

        ts = self._parse_optional_timestamp(memory_time)
        if memory_time and ts is None:
            return (
                "Invalid `memory_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        memory_ts = ts if ts is not None else time.time()

        final_scope, final_group_id, final_platform_id = self._resolve_memory_scope(
            event, group_scope, group_id, platform_id
        )

        cfg = self._cfg()
        embedding_provider = self._resolve_embedding_provider(cfg)
        if not embedding_provider:
            return (
                "No embedding provider is available. "
                "Please configure one in AstrBot provider settings."
            )

        try:
            embedding = await embedding_provider.get_embedding(clean_content)
        except Exception as e:
            return f"Failed to generate embedding: {e}"

        metadata = self._parse_extra_metadata(extra_metadata_json)
        try:
            memory_id = self.memory_rag_store.add_memory(
                content=clean_content,
                embedding=embedding,
                role_ids=role_ids,
                memory_time=memory_ts,
                group_scope=final_scope,
                group_id=final_group_id,
                platform_id=final_platform_id,
                extra_metadata=metadata,
            )
        except Exception as e:
            return f"Failed to write memory: {e}"

        return json.dumps(
            {
                "status": "ok",
                "memory_id": memory_id,
                "memory_time": memory_ts,
                "memory_time_iso": datetime.datetime.fromtimestamp(
                    memory_ts, tz=datetime.timezone.utc
                ).isoformat(),
                "group_scope": final_scope,
                "group_id": final_group_id,
                "platform_id": final_platform_id,
                "related_role_ids": role_ids,
            },
            ensure_ascii=False,
        )

    @llm_tool(name="enhance_memory_rag_read")
    async def memory_rag_read(
        self,
        event: AstrMessageEvent,
        query: str = "",
        related_role_ids: str = "",
        role_match_mode: str = "any",
        start_time: str = "",
        end_time: str = "",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        sort_by: str = "relevance",
        sort_order: str = "desc",
        max_results: int = 10,
        embedding_recall_k: int = 0,
        ignore_group_id: bool = False,
    ) -> str:
        """Read memory records from enhance-mode memory RAG store.

        Args:
            query(string): Optional. Query text for embedding retrieval.
            related_role_ids(string): Optional. Role IDs in JSON array or comma-separated string.
            role_match_mode(string): Optional. `any` or `all`.
            start_time(string): Optional. Start time as unix timestamp or ISO datetime.
            end_time(string): Optional. End time as unix timestamp or ISO datetime.
            group_scope(string): Optional. Full group scope, for example `qq:123456`.
            group_id(string): Optional. Group ID, combined with platform_id when group_scope is empty.
            platform_id(string): Optional. Platform ID used with group_id.
            sort_by(string): Optional. `relevance` or `time`.
            sort_order(string): Optional. `desc` or `asc`.
            max_results(number): Optional. Number of results to return.
            embedding_recall_k(number): Optional. Top-K cutoff for embedding recall before final sorting.
            ignore_group_id(boolean): Optional. When true, do not auto-apply current group filter, allowing cross-group read.
        """
        ready, reason = self._check_memory_rag_ready()
        if not ready:
            return reason
        if self.memory_rag_store is None:
            return "Memory RAG store is not initialized."

        start_ts = self._parse_optional_timestamp(start_time)
        end_ts = self._parse_optional_timestamp(end_time)
        if start_time and start_ts is None:
            return (
                "Invalid `start_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        if end_time and end_ts is None:
            return (
                "Invalid `end_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            return "Invalid time range: `start_time` cannot be greater than `end_time`."

        normalized_ignore_group = (
            ignore_group_id
            if isinstance(ignore_group_id, bool)
            else str(ignore_group_id or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if normalized_ignore_group:
            final_scope = str(group_scope or "").strip()
            final_group_id = str(group_id or "").strip()
            final_platform_id = str(platform_id or "").strip()
            if not final_scope and final_group_id:
                final_scope = (
                    f"{final_platform_id}:{final_group_id}"
                    if final_platform_id
                    else final_group_id
                )
        else:
            final_scope, final_group_id, final_platform_id = self._resolve_memory_scope(
                event, group_scope, group_id, platform_id
            )
        normalized_roles = self._parse_role_ids(related_role_ids)
        normalized_mode = (
            "all" if str(role_match_mode or "").strip().lower() == "all" else "any"
        )
        normalized_sort_by = self._normalize_sort_by(sort_by)
        normalized_sort_order = self._normalize_sort_order(sort_order)

        cfg = self._cfg()
        max_allowed = max(1, cfg.memory_rag.max_return_results)
        try:
            requested_max = int(max_results)
        except (TypeError, ValueError):
            requested_max = 10
        if requested_max <= 0:
            final_max_results = max_allowed
        else:
            final_max_results = min(requested_max, max_allowed)

        try:
            parsed_recall_k = int(embedding_recall_k)
        except (TypeError, ValueError):
            parsed_recall_k = 0
        effective_recall_k = (
            parsed_recall_k
            if parsed_recall_k > 0
            else int(cfg.memory_rag.default_recall_k)
        )

        query_embedding: list[float] | None = None
        clean_query = str(query or "").strip()
        if clean_query:
            embedding_provider = self._resolve_embedding_provider(cfg)
            if not embedding_provider:
                return (
                    "No embedding provider is available. "
                    "Please configure one in AstrBot provider settings."
                )
            try:
                query_embedding = await embedding_provider.get_embedding(clean_query)
            except Exception as e:
                return f"Failed to generate query embedding: {e}"

        try:
            records = self.memory_rag_store.search_memories(
                query_embedding=query_embedding,
                embedding_recall_k=effective_recall_k,
                role_ids=normalized_roles,
                role_match_mode=normalized_mode,
                group_scope=final_scope,
                group_id=final_group_id,
                platform_id=final_platform_id,
                start_time=start_ts,
                end_time=end_ts,
                sort_by=normalized_sort_by,
                sort_order=normalized_sort_order,
                max_results=final_max_results,
            )
        except Exception as e:
            return f"Failed to read memories: {e}"

        return json.dumps(
            {
                "status": "ok",
                "count": len(records),
                "query": clean_query,
                "filters": {
                    "related_role_ids": normalized_roles,
                    "role_match_mode": normalized_mode,
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "ignore_group_id": normalized_ignore_group,
                    "group_scope": final_scope,
                    "group_id": final_group_id,
                    "platform_id": final_platform_id,
                },
                "sort": {"by": normalized_sort_by, "order": normalized_sort_order},
                "max_results": final_max_results,
                "embedding_recall_k": effective_recall_k if clean_query else None,
                "records": records,
            },
            ensure_ascii=False,
        )

    @filter.command_group("enhance")
    def enhance(self) -> None:
        pass

    @permission_type(PermissionType.ADMIN)
    @enhance.command("rag-webui")
    async def rag_webui(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        cfg = self._cfg()
        webui_cfg = cfg.memory_rag_webui
        if not webui_cfg.enable:
            yield event.plain_result(
                "Memory RAG WebUI is disabled.\n"
                "Please enable `memory_rag_webui.enable` in enhance plugin config."
            )
            return

        if self.rag_webui_server is None:
            await self._start_memory_rag_webui()

        if self.rag_webui_server is None:
            yield event.plain_result(
                "Memory RAG WebUI failed to start.\n"
                "Please check AstrBot logs for error details."
            )
            return

        message = (
            "Enhance Memory RAG WebUI\n\n"
            f"URL: {self._memory_rag_webui_url(cfg)}\n"
            "Functions: login, filters, pagination, detail view, delete memory.\n"
        )
        if self.rag_webui_server.password_generated:
            message += (
                "\nPassword is auto generated for this run. "
                "Please check AstrBot logs for the generated password."
            )
        yield event.plain_result(message)

    async def terminate(self) -> None:
        await self._stop_memory_rag_webui()
