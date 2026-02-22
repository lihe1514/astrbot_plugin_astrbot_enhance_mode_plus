# AstrBot Enhance Mode - 群聊增强插件

**版本**: v0.1.0 | **作者**: 阿汐

---

## 功能概述

本插件为 AstrBot 的群聊场景提供增强功能，完全替代内置的「群聊上下文感知」和「主动回复」，并额外支持角色标签和发送者 ID。

- **角色显示**: 在系统提示词中注入用户角色（admin/member），让 LLM 感知发送者权限
- **@ 提及解析**: 解析 LLM 输出中的 `<mention>` 标签，替换为真正的 @ 消息组件
- **引用回复**: 解析 LLM 输出中的 `<quote>` 标签，替换为真正的引用回复组件；同时记录收到的引用消息
- **React 模式**: 将请求改写为"群聊反应模式"（基于群聊历史对新消息做反应）
- **增强群聊上下文**: 以包含发送者 ID、角色标签、消息 ID 的格式记录群聊消息，并注入 LLM 上下文（依赖 React 模式）
- **图片转述**: 使用 LLM 为群聊中的图片生成文字描述，让纯文本模型也能「看到」图片
- **主动回复**: 支持概率触发或模型选择触发（无需被 @），支持白名单限制（依赖 React 模式）

---

## 快速开始

### 安装

将插件文件夹放置于 AstrBot 的 `data/plugins/` 目录下，重启 AstrBot 即可。

### 使用前配置

使用本插件前，请先在 AstrBot 后台**关闭以下内置功能**（避免重复）：

1. **群聊上下文感知**（`group_icl_enable`）→ 关闭
2. **主动回复**（`active_reply.enable`）→ 关闭
3. **引用回复**（`reply_with_quote`）→ 关闭（与本插件的 Quote 功能**互斥**，同时开启会导致模型选择的引用被内置引用覆盖）

以下内置功能**保持开启**：

1. **用户识别**（`identifier`）→ 保持开启（角色显示和发送者 ID 依赖此功能）

然后在插件配置页面启用本插件的对应功能即可。

> **注意**: 如果启用了会话白名单，需要将目标群加入白名单，否则非管理员的群消息会被 pipeline 拦截，无法被本插件记录。

---

## 配置说明

所有配置均可在 AstrBot 控制台的插件配置页面修改。

### 角色显示

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `role_display` | bool | `true` | 在系统提示词的 `<system_reminder>` 中追加 `Role: admin/member` |

启用后，system_reminder 的效果：

```
<system_reminder>User ID: 123456, Nickname: 张三, Role: admin
Group name: 技术交流群
Current datetime: 2026-02-22 21:00 (CST)</system_reminder>
```

### @ 提及解析

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `mention_parse` | bool | `true` | 解析 LLM 输出中的 `<mention id="用户ID">` 标签，替换为 @ 消息组件 |

启用后，LLM 可在输出中使用 `<mention id="123456"> 你好！` 来 @ 群成员。需要同时开启「增强群聊上下文」并包含发送者 ID，否则模型无法获取用户 ID。

### React 模式

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `react_mode.enable` | bool | `false` | 启用后，增强上下文与主动回复能力生效；群聊中的请求会改写为"群聊反应模式" |

启用后，插件在群聊请求中会使用以下模式：

1. 注入群聊历史
2. 将当前消息作为"new message"让模型做即时反应
3. 清空 `req.contexts`，避免与 react 提示词冲突

### 增强群聊上下文

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `group_context.enable` | bool | `false` | 启用增强群聊上下文记录（需先开启 `react_mode.enable`） |
| `group_context.max_messages` | int | `300` | 每个会话保留的最大消息条数 |
| `group_context.include_sender_id` | bool | `true` | 消息格式中包含发送者 ID |
| `group_context.include_role_tag` | bool | `true` | 消息格式中包含 admin/member 角色标签 |
| `group_context.image_caption` | bool | `false` | 使用 LLM 为图片生成文字描述 |
| `group_context.image_caption_provider_id` | string | `""` | 图片转述使用的 LLM 提供商 ID，留空使用默认 |
| `group_context.image_caption_prompt` | string | `"用一句话描述这张图片。"` | 图片转述提示词 |

启用后，群聊消息的记录格式：

```
[张三/123456/21:00:00](admin) #msg78901:  今天天气不错
---
[李四/654321/21:00:05](member) #msg78902:  [Quote 张三: 今天天气不错] 确实
---
[张三/123456/21:00:10](admin) #msg78903:  [Image]
---
[You/21:00:15]: 是的，今天阳光明媚！
```

其中：
- `#msgXXX` 为平台消息 ID，LLM 可通过 `<quote id="XXX">` 引用该消息
- `[Quote 发送者: 内容]` 表示该消息引用了另一条消息
- `[Image]` / `[Image: 描述]` 表示图片（开启图片转述后会附带描述）
- `[At: 昵称]` 表示 @ 某人

### 引用回复（Quote）

插件会自动在 LLM 上下文中注入 Quote 指令，教模型使用 `<quote id="msg_id">` 标签来引用特定消息。模型输出的 `<quote>` 标签会被解析为平台原生的引用回复组件。

使用规则：
- `<quote>` 标签必须出现在输出的**最前面**
- 每条回复只能引用**一条**消息
- `msg_id` 来自聊天记录中 `#msg` 后的数值

> **重要**: 本插件的 Quote 功能与 AstrBot 内置的「引用回复」（`reply_with_quote`）**互斥**。如果同时开启，内置的引用回复会覆盖模型通过 `<quote>` 选择的引用目标。请在 AstrBot 后台关闭 `reply_with_quote`。

### 主动回复

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `active_reply.enable` | bool | `false` | 启用群聊主动回复（需先开启 `react_mode.enable`） |
| `active_reply.mode` | string | `"probability"` | 触发模式：`probability`（概率触发）/`model_choice`（模型选择触发） |
| `active_reply.possibility` | float | `0.1` | `mode=probability` 时，每条消息的回复概率（0.0 - 1.0） |
| `active_reply.model_stack_size` | int | `8` | `mode=model_choice` 时，累计消息达到该长度触发一次模型判定 |
| `active_reply.model_history_messages` | int | `0` | `mode=model_choice` 时，判定阶段额外附带的历史上下文条数（0 表示不附带） |
| `active_reply.model_choice_prompt` | string | 见 schema 默认值 | `mode=model_choice` 时的判定提示词，可使用 `{stack_size}`、`{messages}`、`{history_count}`、`{history_context}`、`{persona_name}`、`{persona_mask}` |
| `active_reply.whitelist` | string | `""` | 限制主动回复的群列表，逗号分隔，留空则所有群生效 |

`model_choice` 模式工作流：

1. 每条群消息进入触发栈
2. 栈长度达到 `model_stack_size` 后，调用模型做一次"是否需要主动回复"判定（判定 prompt 会注入当前生效的人格面具，以及可选历史上下文）
3. 若模型返回 `REPLY`，回复阶段会基于候选消息列表"挑一条感兴趣的消息"进行回复，可使用 Mention / Quote 定向回复对应成员；返回 `SKIP` 则不回复并清空本轮栈

`model_choice` 会输出以下 `info` 级日志：

1. 栈满并开始判定
2. 判定通过（REPLY）
3. 判定拒绝（SKIP 或非标准输出按 SKIP）

---

## 插件结构

```
astrbot_plugin_astrbot_enhance_mode/
├── main.py              # 插件主逻辑
├── metadata.yaml        # 插件元信息
├── _conf_schema.json    # 配置 Schema（WebUI 自动渲染）
└── README.md            # 说明文档
```
