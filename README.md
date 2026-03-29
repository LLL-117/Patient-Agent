# Patient Agent 项目说明

医疗场景下的患者 Agent 后端原型：结构化病历（患者、病例、就诊）、分层记忆（会话、关键事件、画像、向量检索）、多模态问答与可选语音合成；技术栈为 **FastAPI + SQLite**，大模型与向量等能力通过 **阿里云 DashScope** 兼容接口调用。

---

## 一、项目如何启动

### 1. 环境要求

- **Python**：建议 3.10 或 3.11（与当前开发环境一致即可）。
- **操作系统**：Windows / Linux / macOS 均可；下文命令以在项目根目录执行为准。

### 2. 安装依赖

在**项目根目录**执行（依赖列表见根目录 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

说明：

- **FastAPI / Uvicorn / python-multipart / Pydantic**：Web 服务、表单与文件上传、数据校验。
- **dashscope**：CosyVoice（`tts_v2`）与 Qwen TTS 等 SDK 调用；对话 / 向量 / 多模态主链路走兼容 HTTP，未配置 API Key 时部分能力会降级或返回明确错误，但服务仍可启动。

语音合成若走 CosyVoice，需 **dashscope** 版本满足 `dashscope.audio.tts_v2`（`requirements.txt` 中已设下限）。

### 3. 环境变量（按需）

| 变量 | 作用 |
|------|------|
| `QWEN_API_KEY` | 主 API Key（与下方二选一或共用） |
| `DASHSCOPE_API_KEY` | 部分接口优先读取；未设时多处回退到 `QWEN_API_KEY` |
| `LOG_LEVEL` | 日志级别，默认 `INFO` |
| `LOG_HTTP` | 设为 `1` 时打印 HTTP 访问日志 |
| `TTS_ENABLED` | 设为 `false` / `0` 等可全局关闭服务端 TTS |
| `QWEN_TTS_MODEL` / `COSYVOICE_TTS_MODEL` 等 | 详见代码内默认值与注释 |
| `MEMORY_USAGE_EXTRACT_ENABLED` | 默认开启；`0`/`false` 关闭「**按消息量**将短期会话抽成长期记忆」 |
| `MEMORY_USAGE_EXTRACT_THRESHOLD` | 默认 `20`；自上次用量抽取以来，session 内**新增消息条数**达到该值则触发一次抽取（需 `QWEN_API_KEY`） |
| `MEMORY_PERIODIC_EXTRACT_ENABLED` | 默认 `false`；`1`/`true` 开启后台**定时**扫描各会话并抽取 |
| `MEMORY_PERIODIC_EXTRACT_INTERVAL_SECONDS` | 默认 `86400`（秒）；定时任务每轮间隔，且同一会话两次定时抽取至少相隔该间隔 |
| `MEMORY_PERIODIC_EXTRACT_MIN_MESSAGES` | 默认 `2`；定时任务仅处理消息数 ≥ 该值的会话 |
| `MEMORY_EXTRACT_DEBOUNCE_SECONDS` | 默认 `120`；同一会话在窗口内不重复抽取（用量与定时共用，避免短时间重复调模型） |

**短期 → 长期自动抽取（与手动 `extract-dialogue` 同源）**

- **用量策略**（默认开）：`query-multimodal` 在本轮写入 user/assistant 后，若当前 session 自上次用量抽取以来**新增消息条数** ≥ `MEMORY_USAGE_EXTRACT_THRESHOLD`，则异步调用 `run_extract_from_dialogue` + `persist_extraction_result`（需 `QWEN_API_KEY`）。状态记在表 `memory_session_extract_state`。
- **定时策略**（默认关）：`MEMORY_PERIODIC_EXTRACT_ENABLED=true` 时，`main` 启动后台协程，每隔 `MEMORY_PERIODIC_EXTRACT_INTERVAL_SECONDS` 扫描 `session_memory`，对满足「消息数 ≥ 最小值且距上次定时抽取已满间隔」的会话做同样抽取。启动日志会出现 `periodic memory extract started`；未开启时会出现 `periodic memory extract disabled`，属预期行为。

数据库路径由 `patient_db` 决定：默认在**项目根目录**下自动创建 `data/`，首次访问时建库（含 `memory_session_extract_state` 等表）。

### 4. 启动主应用（必选）

在**仓库根目录**（包含 `backend` 包与本文 `README.md` 的目录）执行：

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

常用验证：

- 健康检查：<http://127.0.0.1:8000/health>，应返回 `{"status":"ok"}`。
- OpenAPI 文档：<http://127.0.0.1:8000/docs>。
- 静态联调页（挂载在 `/test`）：例如统一入口 <http://127.0.0.1:8000/test/patient_app.html>。

### 5. 可选：独立 MCP 服务

若需单独调试 MCP（**Streamable HTTP**：`POST /mcp` JSON-RPC、`application/json` 响应，**非** stdio/SSE；旧版 REST 仍可用）或 invoke、ReAct 等，可在**同一仓库根目录**另起进程（端口示例 `8100`）：

```bash
python -m uvicorn backend.mcp_server:app --reload --host 127.0.0.1 --port 8100
```

同进程还提供：**`POST /mcp`**（JSON-RPC Streamable HTTP，响应 `application/json`）、**`GET /mcp`**→405、**`DELETE /mcp`**（带 `MCP-Session-Id` 注销会话）；旧版 **`/mcp/tools`、`/mcp/invoke`、agent-call、agent-react** 仍可用。详细握手与示例见 `backend/MCP_QUICKSTART.md`。

### 6. 可选：种子数据

若仓库内提供 `seed_data.py`、`batch_seed_all_patients.py` 等脚本，可在配置好 Python 路径后于根目录执行（具体参数以脚本内说明为准），用于写入演示患者与就诊数据。

---

## 二、项目目录结构

### 2.1 仓库根目录一览

```text
<项目根目录>/                    # 启动 uvicorn 时的工作目录（须能 import backend）
├── README.md                    # 项目说明（启动方式 + 目录结构）
├── requirements.txt             # Python 依赖（pip install -r requirements.txt）
├── tetris.html                  # 与主项目无关的独立静态页（若有）；可忽略
├── assets/                      # 可选静态资源（如联调用测试图片）
│
├── backend/                     # Python 包：全部业务与 Agent 代码
│   ├── __init__.py              # 包标识
│   ├── main.py                  # FastAPI 主应用
│   ├── agent_module.py
│   ├── patient_db.py
│   ├── memory_extract.py
│   ├── memory_refresh.py        # 用量/定时：短期会话 → 长期记忆自动抽取
│   ├── memory_vector.py
│   ├── session_media.py
│   ├── react_api.py
│   ├── react_planner.py
│   ├── mcp_server.py
│   ├── mcp_streamable_http.py   # MCP 规范 Streamable HTTP（JSON-RPC 单端点，非 SSE）
│   ├── seed_data.py
│   ├── batch_seed_all_patients.py
│   ├── convert_batch_data_to_cn.py
│   ├── convert_raw_json_plan_to_cn.py
│   ├── PROJECT_INTERVIEW_GUIDE.md
│   ├── MCP_QUICKSTART.md
│   ├── static/                  # 浏览器联调页（挂载路径 /test）
│   │   ├── patient_app.html
│   │   ├── patient_query.html
│   │   ├── patient_chat.html
│   │   ├── agent_query.html
│   │   ├── patient_test.html
│   │   ├── memory_chat_test.html
│   │   └── session_memory_view.html
│   ├── audio_cache/             # 【运行时生成】TTS 生成的音频文件
│   └── session_image_cache/     # 【运行时生成】会话内上传图片落盘
│
└── data/                        # 【运行时生成】SQLite 等业务数据目录
    └── patient_agent.db         # 默认数据库文件名（首次访问时创建）
```

> **路径约定**：`patient_db` 将库文件放在**项目根目录**下的 `data/patient_agent.db`；`agent_module` 将 TTS 与上传图放在 `backend/audio_cache`、`backend/session_image_cache`。

---

### 2.2 `backend/` 核心 Python 模块

| 文件 | 职责摘要 |
|------|-----------|
| `main.py` | 创建 `FastAPI` 实例；注册中间件；挂载 `StaticFiles`（`/test` → `backend/static`）；注册 `agent_router`、`react_router`；患者/病例/就诊与记忆等 REST；**`startup`** 调用 **`start_periodic_refresh_background(db)`**，**`shutdown`** 调用 **`stop_periodic_refresh_background`**（可选定时短期→长期抽取）。 |
| `agent_module.py` | Agent 路由：`POST /api/agent/query-multimodal`（文本+可选图+`tts_voice`）；**纯问候**先匹配（`tool_name=agent.greeting`，不查库、不拼病历记忆块；含图时关闭问候短路）；**身份**：`patient_code` 与 `phone` 可只填其一，**若两者都填则须与库中为同一患者**（手机号按 11 位数字规范化比对）；失败时 HTTP 400/404 的 `detail` 为固定句「你输入的编号或手机号有误。」（未带任何身份且需查病历时仍返回 `identity.request_verification` 话术）。会话写入成功后调用 **`schedule_usage_refresh_if_needed`**（见 `memory_refresh`）。病历检索与 `_merge_memory_into_response`；Qwen 多模态与 Qwen-TTS / CosyVoice；`GET /api/agent/session-image/{id}`、`GET /api/agent/audio/{id}`。 |
| `patient_db.py` | `PatientDatabase`：SQLite 与 `memory_*` 全表；**`memory_session_extract_state`**（按 session 记录用量/定时抽取进度）；`get_patient` / **`get_patient_by_phone`**；**`list_session_memory_rows`**（供定时任务扫描）；用户画像 **`normalize_user_profile_keys`**。 |
| `memory_extract.py` | 调用 DashScope 对「对话」或「业务摘要」做 JSON 抽取：关键事件 + 用户画像；`persist_extraction_result` 统一落库；供 `main` 与自动刷新使用。 |
| `memory_refresh.py` | **用量阈值**与**定时**将短期会话抽成长期记忆（与 `extract-dialogue` 同源）；由 `query-multimodal` 写入会话后调度，及 `main` 启动可选后台任务。 |
| `memory_vector.py` | 文本 embedding、关键事件入向量表；FTS5 全文；`hybrid_search` 等混合检索，供 Agent 上下文组装使用。 |
| `session_media.py` | 拼装会话消息中的 Markdown 后缀（图片链接、语音链接等），避免正文塞满 base64。 |
| `react_api.py` | 对外暴露 ReAct / 规划类路由（内部复用 `mcp_server.MCPToolbox` 等）。 |
| `react_planner.py` | ReAct 规划与自洽等多步推理逻辑（HTTP 由 `react_api` 转发）。 |
| `mcp_server.py` | 独立 FastAPI：**`POST /mcp`** 为 MCP **Streamable HTTP**（JSON-RPC：`initialize`、`tools/list`、`tools/call` 等；响应 **application/json**，非 SSE）；`GET /mcp`→405；另保留 `GET /mcp/tools`、`POST /mcp/invoke`、`agent-call` / `agent-react`。 |
| `mcp_streamable_http.py` | Streamable HTTP 的 JSON-RPC 分发、会话头 `MCP-Session-Id`、Origin 校验。 |
| `seed_data.py` | 单患者或演示数据写入脚本入口（按脚本内用法执行）。 |
| `batch_seed_all_patients.py` | 批量导入/种子患者数据脚本。 |
| `convert_batch_data_to_cn.py` | 批量数据转中文/结构化落库辅助脚本。 |
| `convert_raw_json_plan_to_cn.py` | 原始 JSON 计划转中文等转换脚本。 |

---

### 2.3 `backend/static/` 静态联调页

主服务启动后，浏览器访问 **`http://<host>:<port>/test/<文件名>`**（注意 `main` 将静态目录挂在 `/test`，无默认 `index.html` 时需写全文件名）。

| 文件 | 用途简述 |
|------|-----------|
| `patient_app.html` | **推荐主入口**：侧栏切换「Query 页面 / 聊天页面」；患者编号、手机号、会话 ID、声音选择（无=不播报）；调用 `query-multimodal`。 |
| `patient_query.html` | 仅自然语言 Query + 可选图；Planner 调试可选。 |
| `patient_chat.html` | 仅多轮聊天；顶栏配置与主应用类似。 |
| `agent_query.html` | 精简版多模态请求页，字段较少。 |
| `patient_test.html` | **纯 REST 联调**：Upsert 患者、写病例、写就诊、按 ID/编号读回；不经过 Agent。 |
| `memory_chat_test.html` | 最小请求体调用 `query-multimodal`，便于看返回 JSON。 |
| `session_memory_view.html` | 查看/渲染会话记忆、语音与图片引用等（按页面实现为准）。 |

---

### 2.4 运行时与数据目录

| 路径 | 内容 |
|------|------|
| `data/patient_agent.db` | 默认 SQLite：患者主数据、病例、就诊、`session_memory`、**`memory_session_extract_state`**、关键事件、画像、向量、FTS 等；**删除前请备份**。 |
| `backend/audio_cache/` | TTS 返回给前端的音频以文件形式缓存，通过 `/api/agent/audio/{audio_id}` 提供。 |
| `backend/session_image_cache/` | 用户经多模态表单上传的图片落盘，会话正文内用短链引用。 |

---

### 2.5 文档与其它

| 路径 | 说明 |
|------|------|
| `backend/PROJECT_INTERVIEW_GUIDE.md` | 架构、记忆分层、接口索引、面试表述等**长篇说明**，适合深入阅读。 |
| `backend/MCP_QUICKSTART.md` | 独立 MCP 进程启动、工具列表与调用示例。 |
| 根目录 `.gitignore` | 已忽略 `.venv`、`data/*.db`、`backend/audio_cache/`、`backend/session_image_cache/`、`.env`、`.idea/` 等；提交前勿把密钥与本地库文件推进仓库。 |

---

### 2.6 小结

- **无前端构建链路**：联调依赖 `backend/static` 下 HTML + 浏览器直连 API。
- **单进程主服务**即可覆盖病历 CRUD + Agent + 记忆；**MCP** 为可选第二进程。
- 静态页中 **音色** 对应表单字段 **`tts_voice`**：传 **`none` 或不传** 表示不播报（无独立 `tts_enabled` 开关）。
- **短期→长期**：用量阈值（默认）+ 可选定时后台（默认关）；详见上文环境变量与 `memory_refresh.py`。
- **独立 MCP 进程**：优先使用规范 **`POST /mcp`**（`mcp_streamable_http.py`）；旧 REST 为兼容保留。
- 更细的 **REST 路径列表、记忆表字段、环境变量全集** 以 `backend/PROJECT_INTERVIEW_GUIDE.md` 与源码为准。
