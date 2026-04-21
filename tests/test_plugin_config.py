from __future__ import annotations

import math

from astrbot_plugin_astrbot_enhance_mode.plugin_config import parse_plugin_config


def test_parse_plugin_config_defaults() -> None:
    cfg = parse_plugin_config(None)
    assert cfg.group_history.enable is False
    assert cfg.active_reply.enable is False
    assert cfg.active_reply.mode == "probability"
    assert cfg.active_reply.model_choice_provider_id == ""
    assert math.isclose(cfg.active_reply.possibility, 0.1)
    assert cfg.group_history_enabled is False
    assert cfg.active_reply_enabled is False
    assert cfg.web_search.request_mode == "auto"
    assert cfg.web_search.base_url_override == ""
    assert cfg.active_reply.auto_create_session is False
    assert cfg.active_reply.auto_session_title == "主动回复-{group_id}"


def test_probability_is_clamped_and_nan_falls_back() -> None:
    cfg_high = parse_plugin_config(
        {
            "group_features": {"react_mode_enable": True},
            "active_reply": {"enable": True, "possibility": 9},
        }
    )
    assert math.isclose(cfg_high.active_reply.possibility, 1.0)
    assert cfg_high.active_reply_enabled is True

    cfg_low = parse_plugin_config({"active_reply": {"possibility": -0.5}})
    assert math.isclose(cfg_low.active_reply.possibility, 0.05)

    cfg_nan = parse_plugin_config({"active_reply": {"possibility": "nan"}})
    assert math.isclose(cfg_nan.active_reply.possibility, 0.1)


def test_active_reply_mode_and_limits_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "active_reply": {
                "mode": "something_else",
                "model_stack_size": 0,
                "model_history_messages": -99,
                "model_choice_provider_id": "  provider-1  ",
                "whitelist": "a,b, c",
            },
            "global_settings": {
                "lru_cache": {"max_origins": 0},
                "timeouts": {"image_caption_sec": -1, "model_choice_sec": "0"},
            },
        }
    )

    assert cfg.active_reply.mode == "probability"
    assert cfg.active_reply.model_stack_size == 1
    assert cfg.active_reply.model_history_messages == 0
    assert cfg.active_reply.model_choice_provider_id == "provider-1"
    assert cfg.active_reply.whitelist == ["a", "b", "c"]
    assert cfg.global_settings.lru_cache.max_origins == 1
    assert math.isclose(cfg.global_settings.timeouts.image_caption_sec, 45.0)
    assert math.isclose(cfg.global_settings.timeouts.model_choice_sec, 45.0)


def test_web_search_request_mode_and_base_override_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "web_search": {
                "request_mode": "RESPONSES",
                "base_url_override": "  https://example.com/custom/v1  ",
            }
        }
    )
    assert cfg.web_search.request_mode == "responses"
    assert cfg.web_search.base_url_override == "https://example.com/custom/v1"

    cfg_invalid = parse_plugin_config({"web_search": {"request_mode": "unknown"}})
    assert cfg_invalid.web_search.request_mode == "auto"


def test_active_reply_auto_create_session_parsing() -> None:
    cfg_enabled = parse_plugin_config(
        {
            "active_reply": {
                "auto_create_session": True,
                "auto_session_title": "自定义标题-{group_id}",
            }
        }
    )
    assert cfg_enabled.active_reply.auto_create_session is True
    assert cfg_enabled.active_reply.auto_session_title == "自定义标题-{group_id}"

    cfg_disabled = parse_plugin_config({"active_reply": {"auto_create_session": False}})
    assert cfg_disabled.active_reply.auto_create_session is False

    cfg_empty_title = parse_plugin_config(
        {"active_reply": {"auto_session_title": ""}}
    )
    assert cfg_empty_title.active_reply.auto_session_title == "主动回复-{group_id}"
