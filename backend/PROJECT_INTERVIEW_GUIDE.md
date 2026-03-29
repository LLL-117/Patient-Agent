# Patient Agent 项目 · 面试复习手册

面向当前 `backend` 代码整理，便于面试前复习。技术栈：FastAPI、SQLite、阿里云 DashScope 兼容 OpenAI API。

---

## 1. 项目定位

一句话：面向医疗场景的后端服务，把患者主数据、病例与就诊等业务数据，和对话记忆、抽取式长期记忆、多模态问答放在同一套 API 里；无登录场景下可用 `patient_code` 和/或手机号定位患者（**两者同时提交时必须指向库中同一患者**），并通过大模型完成信息抽取、语义检索、多模态理解与语音合成。

能力边界（设计意图）：

- 不是完整互联网医院 HIS，而是「Agent + 记忆 + 结构化病历查询」的演示／原型级后端。
- 强调可解释的数据落库（关键事件、画像、会话）与可替换的模型供应商（兼容模式 Base URL + API Key）。

对应 API：探活 `GET /health`；全量业务与记忆接口的完整路径见下文第 15 节。

> 面试表达：这是一个医疗方向的 Patient Agent 后端，核心是结构化病历存储、分层记忆、多模态问答；业务上解决「同一患者多轮对话里能带上既往病历和长期记忆」；工程上把抽取、向量、全文、工具规划模块化，方便演示和扩展。

---

## 2. 整体架构

主进程是 FastAPI：`backend/main.py` 挂载静态页 `/test`，注册 `agent_router` 与 `react_router`，其余为患者、病例、就诊、记忆等 REST 接口。数据层是单例 `PatientDatabase`，SQLite 默认路径 `data/patient_agent.db`。

可选：单独启动 `mcp_server.py`（`uvicorn backend.mcp_server:app`），提供 MCP 风格工具列表、invoke、轻量 QwenRouter、多步 agent-react；与主应用共用同一套数据模型思路，进程独立。

外部依赖：DashScope 兼容端点（Chat、Embeddings、VL、TTS 等）。

下面示意图便于脑补模块关系（不要求背 ASCII）：

```
FastAPI main → CRUD（患者/病例/就诊/QA）+ Memory（抽取/向量/FTS）+ Router（agent、react_plan）
                        ↓
                 PatientDatabase (SQLite)

可选：mcp_server 独立进程 → MCP 工具 / invoke / QwenRouter / agent-react
```

对应 API（主应用，根路径为服务地址，如 `http://127.0.0.1:8000`）：静态页 `GET /test/`（挂载 `backend/static`）；其余 REST 与 Agent 见第 15 节。对应 API（独立 MCP 进程，`uvicorn backend.mcp_server:app`）：`GET /health`、`GET /mcp/tools`、`POST /mcp/invoke`、`POST /mcp/agent-call`、`POST /mcp/agent-react`。

> 面试表达：架构上是单体 FastAPI + 单库 SQLite；Agent 和 Planner 用 APIRouter 插拔；MCP 可单独起服务做工具演示，与主应用共享数据模型思想，但不和主进程绑死。

---

## 3. 记忆架构（分层）

短期会话：表 `session_memory`，存 `messages` JSON 数组（user/assistant/system，含时间戳，可选 extras）。按 `session_id` 与 `patient_id` 区分；可整段删除会话。

长期关键事件：表 `memory_key_events`，含标题、摘要、来源、日期、置信度、raw_json；由业务抽取或对话抽取写入，列表按时间倒序。

融合用户画像：表 `memory_user_profile`，`profile_json` 键值对；抽取结果会 merge 进已有画像。落库与展示侧通过 `patient_db.normalize_user_profile_keys` 将常见**英文键转为中文键**（与 `memory_extract` 抽取 schema 一致），避免回答里混用 `chronic_focus` 等英文键名。

对话偏好：表 `memory_settings`，`preferences` JSON；与画像分离，按患者维度配置。

向量（稠密）：表 `memory_vector_chunks`，含 `embedding_json`、`content_text`、`source_type` 与 `source_id`；关键事件文本做 embedding；向量存在 SQLite 里，不是独立向量库。

全文（稀疏）：虚拟表 `memory_key_events_fts`（FTS5），与关键事件同步的 `body`；用于 bm25 排序与混合检索。

对应 API：单表无独立 REST；读写分散在以下接口——偏好 `GET /patients/{patient_id}/memory/settings`、`PATCH /patients/{patient_id}/memory/settings`；会话 `GET /patients/by-code/{patient_code}/memory/session/{session_id}`、`DELETE` 同路径；抽取与向量、画像汇总见第 5 节与第 15 节。

> 面试表达：记忆拆成短期会话和长期资产；长期包括结构化关键事件、融合画像、偏好。检索用 SQLite 存向量 JSON + FTS5 全文，查询时做稠密与稀疏的 RRF 混合，不引入额外中间件也能讲清权衡。

---

## 4. 短期记忆的计数方案

存储：`session_memory` 以 `session_id` 为主键，字段含 `patient_id`、`messages`（整条会话一个 JSON 数组）、时间戳。单条消息含 `role`、`content`、`ts`，可选 `extras`（多模态元数据等）。

写入：已去掉「单独 HTTP 追加会话消息」接口。短期会话只在调用 `POST /api/agent/query-multimodal` 且传入 `session_id` 时，在本轮推理结束后依次写入 user 与 assistant。用户上传图片会落到 `session_image_cache`，正文里用短链 `/api/agent/session-image/{image_id}` 引用。

读取：`GET /patients/by-code/{patient_code}/memory/session/{session_id}`，`newest_first=True` 时按时间倒序展示消息。删除会话：`DELETE /patients/by-code/{patient_code}/memory/session/{session_id}`。

写入短期消息：仅通过 `POST /api/agent/query-multimodal` 传 `session_id`，本轮结束后写入 user/assistant；无单独 append 接口。取会话内引用的本地图片：`GET /api/agent/session-image/{image_id}`。

> 面试表达：短期记忆是会话级 JSON 追加，与主问答绑在同一次多模态请求里；多模态用 extras 和本地图片短链控制体积。

---

## 5. 长期记忆的生成与存储

### 生成（抽取）

业务抽取：`POST /patients/by-code/{patient_code}/memory/extract-business` 读病例与就诊摘要，经 `memory_extract.run_extract_from_business`，LLM 输出 `key_events` 与 `user_profile`。

对话抽取：`POST /patients/by-code/{patient_code}/memory/extract-dialogue`（body：`session_id`）读 `session_memory`，经 `run_extract_from_dialogue`，同样输出事件与画像。

落库：`insert_key_events` 写入 `memory_key_events`，并往 `memory_key_events_fts` 插一行（body 为标题+摘要）；`merge_extracted_user_profile` 更新画像。

### 向量化与索引

`index_key_events` 对关键事件文本调 `/embeddings`，写入 `memory_vector_chunks`（`source_type=key_event`）。`reindex` 会先同步 FTS 再全量重建向量（有条数上限，如 500）。

### 检索

稠密侧：查询与文档向量余弦相似。稀疏侧：FTS5 的 MATCH 与 bm25。混合：两路排序用 RRF 融合（`MEMORY_HYBRID_RRF_K` 等可调）。

对应 API：业务抽取 `POST /patients/by-code/{patient_code}/memory/extract-business`（无 body，依赖库内病例与就诊）。对话抽取 `POST /patients/by-code/{patient_code}/memory/extract-dialogue`（JSON body：`session_id`）。向量与索引 `POST /patients/by-code/{patient_code}/memory/vector`（JSON：`operation` 为 `search` 或 `reindex`；`search` 时必填 `query`，可选 `top_k`、`search_mode` 为 `hybrid` | `vector` | `fts`）。已抽取汇总 `GET /patients/{patient_id}/memory/extracted`（Query：`key_events_limit`、`key_events_offset`）。

> 面试表达：长期记忆是抽取落库 + 双索引；检索默认混合 RRF；没有上 Chroma 这类独立向量库，是 SQLite 一体化的取舍。

---

## 6. MCP 与工具调用能力

第一套：`mcp_server.py` 独立 FastAPI。`MCPToolbox` 提供 `auth.verify_identity`、`case.query_cases`、`visit.query_visits`。`POST /mcp/invoke` 直接调工具。`POST /mcp/agent-call` 走 `QwenRouter.route()` 单次 JSON 规划再 invoke；QwenRouter 多采样，按 tool_name 多数票与置信度，自我一致性（单步）。`POST /mcp/agent-react` 是多轮 ReAct。

第二套：主应用里 `react_api` 复用同一 `MCPToolbox`（从 `mcp_server` import），共享工具实现。

与主 Agent 的关系：主对话 `agent_module._run_text_query` 用本地关键词路由 `_route_query`，不是 MCP 的 QwenRouter；工具语义对齐，调用链分开。

对应 API（须在 MCP 独立服务上调用，非 `/api/agent` 同端口）：`GET /health`、`GET /mcp/tools`、`POST /mcp/invoke`（body：`tool_name`、`arguments`）、`POST /mcp/agent-call`（body：`user_query`、`context`）、`POST /mcp/agent-react`（body：`user_query`、`context`）。主应用侧多步规划见 `POST /api/agent/react-plan`（第 8 节）。

> 面试表达：MCP 是工具注册表、手动 invoke、可选 LLM 路由；路由上支持单步 Qwen JSON 规划与多采样投票。主 Agent 仍用轻量规则路由，与 MCP 演示共用 Toolbox 实现，但刻意解耦。

---

## 7. 多模态交互能力

入口：`POST /api/agent/query-multimodal`，`multipart/form-data`。

**问候短路**：`_run_text_query` 开头若 `_is_greeting_query(query)` 为真（短句问候、无业务词、正文不含患者编号/手机；文本先做零宽/BOM 清理与 NFC），直接返回 `tool_name=agent.greeting` 与固定欢迎话术，**不查库、不拼**「病历与记录检索」记忆块。本轮**带本地文件或 URL 图片**时 `allow_greeting=False`，不走问候分支。

**身份解析**（非问候之后）：`patient_code`、`phone` 可从表单或 `query` 正文解析；`patient_db.get_patient_by_phone` 按院内手机（去空格/符号后比对 11 位）查找。**仅填其一**时按编号或手机解析患者；**两者都填**时必须对应**同一** `patient_id`，否则 **HTTP 400**（或查无此人时 **404**），`detail` 统一为：**「你输入的编号或手机号有误。」**（手机号格式非法同样使用该句。）未带任何身份且意图为查病历/病例/验证时，仍返回 JSON 体 `identity.request_verification`，而非上述 HTTP 错误。

文本：表单字段 `query`。公网图片 URL：`image_url`，经 `_sanitize_public_image_url` 过滤内网与本地。本地上传：`image_file` 转 base64 data URL，可选落盘 `session_image_cache`。视觉理解：`_build_qwen_multimodal_answer` 调兼容模式 `/chat/completions` 与 `qwen-vl-*`，content 为多段 text + image。Prompt：系统提示 + 用户模板（含用户问题、记忆上下文、病历检索 core、compact_data）。语音：表单项 **`tts_voice`** 控制音色；传 **`none` 或不传** 则不请求 TTS（无独立 `tts_enabled` 表单开关；环境变量 `TTS_ENABLED` 仍可全局关闭）。成功时音频落 `audio_cache`，返回 `/api/agent/audio/{id}`。降级：VL 失败则回退纯文本检索并提示。

对应 API：`POST /api/agent/query-multimodal`（`multipart/form-data`：`query` 必填；可选 `patient_code`、`phone`、`session_id`、`tts_voice`、`image_url`、`image_file`）。取图：`GET /api/agent/session-image/{image_id}`。取语音：`GET /api/agent/audio/{audio_id}`。

> 面试表达：多模态是 OpenAI 式 messages + image_url 或 data URL；安全上限制非公网图 URL；检索先走结构化病历，再让 VL 在带记忆上下文的模板里看图。身份上双字段交叉校验避免「编号+他人手机」误查；问候与业务查询分流降低误触工具。

---

## 8. Planner（ReAct、CoT、自我一致性）

模块：`backend/react_planner.py`。HTTP：`POST /api/agent/react-plan`（`react_api`）。

对应 API：`POST /api/agent/react-plan`（JSON：`user_query`；可选 `context`、`max_steps`（1～12）、`sc_runs`（1～8）、`include_all_episodes`）。与 MCP 侧 `POST /mcp/agent-react`（JSON：`user_query`、`context`）逻辑同类，入口与端口不同。

单条轨迹：每步 LLM 输出 JSON，含 `thought`、`cot_steps`（CoT）、`action`（工具名或 finish）、`arguments`、`confidence`。非 finish 则 invoke，把 observation 写入 `prior_trace`，直到 finish 或达到 `QWEN_REACT_MAX_STEPS`。

自我一致性：并行跑 `QWEN_REACT_SC_RUNS` 条完整轨迹；对工具调用链 fingerprint 多数投票，胜组内取平均置信度最高的轨迹。

与 QwenRouter 对比：QwenRouter（mcp agent-call）是单步选工具，一致性对 tool_name 投票。react_planner（react-plan）是多步 ReAct，一致性对整条工具链投票。

> 面试表达：Planner 分两层：轻量单步给 MCP demo；重规划用 ReAct、CoT、多轨迹投票，适合多工具串联。面试可强调：单步投 tool，多步投轨迹 fingerprint。

---

## 9. 插拔式 API 与模块化

Agent：`agent_module.router` 通过 `app.include_router(agent_router)` 挂载。Planner：`react_api.router`，标签 Planner。MCP：独立 `mcp_server.py` 可单独部署。数据层：`PatientDatabase` 集中；记忆与向量在 `memory_vector.py`、`memory_extract.py`。环境变量：`QWEN_*`、`QWEN_EMBEDDING_MODEL`、`MEMORY_HYBRID_*`、`LOG_LEVEL`、`LOG_HTTP` 等。

对应 API：`main.py` 中 `app.include_router(agent_router)` 无前缀，故 Agent 路径为 `/api/agent/...`；`app.include_router(react_router)` 提供 `POST /api/agent/react-plan`。患者与记忆 CRUD 见 `main.py` 中 `@app` 路由，前缀 `/patients/...`。

> 面试表达：用 FastAPI Router 拆模块，数据访问统一 PatientDatabase；模型相关环境变量注入。MCP 独立进程可选，主 API 不依赖它也能跑通核心问答。

---

## 10. Query 的完整执行流程（主路径：多模态问答）

以下以 `POST /api/agent/query-multimodal` 为例（可带 `patient_code`、`phone`、`session_id`、`tts_voice`、图）。

第一步：解析表单中的 `query`、`patient_code`、`phone`、`session_id`、`tts_voice`、`image_url` 或 `image_file`。

第二步：进入 `_run_multimodal`，内部调用 `_run_text_query`（无图时 `allow_greeting=True`）：若命中**纯问候**，直接返回 `agent.greeting`，结束文本路径。

第三步（非问候）：合并表单与正文中的 `patient_code` / `phone`；**`_resolve_patient_identity`** 做交叉校验（双字段须同患者，否则 HTTP 错误，`detail` 为「你输入的编号或手机号有误。」）；仅单一标识时按库查询。若无患者且意图需要身份，返回 `identity.request_verification` 的 JSON 应答（非 HTTP 异常）。

第四步：`_route_query` 得 `case.query_cases` / `visit.query_visits` / `auth.verify_identity`；`list_medical_cases` 或 `list_visit_records`；`_build_detailed_summary`；`_merge_memory_into_response` 中 `_build_memory_context_block`（画像、偏好、同 session、`hybrid_search` RRF）；data 含 `retrieval_answer_core`、`memory_context`。

第五步：若有图，`_build_qwen_multimodal_answer`（VL），输入含记忆与 core。

第六步：若 `tts_voice` 请求合成且环境允许，TTS 写 `audio_cache`，返回 `audio_url`。

第七步：若带 `session_id` 且本轮成功解析到 `patient_id`，依次写入 user / assistant 短期消息。

无图时：直接返回合并后的 `answer`（含记忆块；问候时无记忆块）。

对应 API：对外仅 `POST /api/agent/query-multimodal` 一条主入口；内部 `_run_text_query` / `_run_multimodal` 无额外 HTTP。

> 面试表达：问候与业务分流 → 身份交叉校验 → 结构化检索 → 记忆增强 → 多模态（可选）→ 会话落库；记忆块显式拼接；混合检索两路可解释。

---

## 11. 当前最值得强调的工程点

1. 分层记忆：短期 JSON 会话与长期事件、画像、偏好职责清晰。  
2. 混合检索：SQLite FTS5 与 embedding JSON，RRF 融合，无额外向量库运维。  
3. 抽取与落库闭环：抽取后关键事件、FTS、向量索引（失败不阻塞主流程）。  
4. 多模态与记忆对齐：检索 core 与记忆上下文分离进 VL 模板。  
5. 双 Planner：单步投票与多步 ReAct，按场景选复杂度。  
6. 无登录身份：`patient_code`+`phone` 双字段交叉校验与统一错误文案，降低误查他人病历风险。  
7. 可观测性：`LOG_LEVEL`、`LOG_HTTP`、关键路径 logging。

对应 API：与上述能力相关的 HTTP 仍以第 7、8、15 节为准；探活 `GET /health`。

> 面试表达：若只能讲三点：分层记忆、混合检索、抽取落库闭环；多模态和 Planner 是加分项。

---

## 12. 面试短答模板

问：向量存在哪？  
答：存在 SQLite 表 `memory_vector_chunks` 的 `embedding_json` 字段里，不是独立向量库；检索在应用层算余弦。需要时可迁到 Chroma、Milvus 等。

问：混合检索怎么做？  
答：一路 FTS5 bm25，一路 embedding 余弦，两路各出排序，用 RRF 融合，再按 `event_id` 去重取正文。

问：MCP 和主 Agent 什么关系？  
答：工具实现可复用 `MCPToolbox`；主 Agent 是轻量规则加数据库查询；MCP 服务侧重 LLM 选工具与多步规划演示。

问：短期记忆怎么和问答一致？  
答：写入在本轮推理成功之后，且 `session_id` 显式传入，避免「只写库、不经过模型」的旁路。

问：编号和手机都填错了会怎样？  
答：身份解析失败时 HTTP **400/404**，响应体 `detail` 为 **「你输入的编号或手机号有误。」**；与 `auth.verify_identity` 未通过时返回给前端的 `answer` 文案一致。

---

## 13. 当前版本的边界

无登录鉴权、RBAC、审计日志；无生产级多租户隔离。向量检索无 ANN 索引，数据量大时全表扫向量是瓶颈。中文 FTS 用 `unicode61`，不是专业中文分词。Agent 路由是关键词规则，不是全程 LLM 规划。reindex 有条数上限（如 500）。MCP 与 main 双进程时需注意 SQLite 并发写。

> 面试表达：主动说明这是原型边界；SQLite 混合检索适合演示与中小数据；上生产会考虑向量库、权限、异步索引队列。

---

## 14. 最后一句话总结（约 30 秒）

这是一个 FastAPI 医疗 Agent 后端：用 SQLite 统一管理患者、病例、就诊与分层记忆；长期记忆通过对话或业务抽取写入关键事件与画像，并用 FTS5 与 DashScope Embedding 做混合检索；对外提供多模态问答（VL+TTS）与可选的 ReAct Planner；工程上强调模块化路由、环境变量换模型、可解释的检索与记忆拼接。

---

## 15. 接口白话说明（无表，按类罗列）

下列均为「主应用」路径（默认 `http://127.0.0.1:8000`，以实际部署为准）。花括号内为路径参数。

系统与健康：`GET /health` 探活。

患者：`POST /patients/upsert` 创建或更新患者。`GET /patients/{patient_id}` 按 `patient_id` 查。`GET /patients/by-code/{patient_code}` 按编号查。`POST /patients/{patient_id}/cases` 新增病例；`GET /patients/{patient_id}/cases` 列表。`POST /patients/{patient_id}/visits` 新增就诊；`GET /patients/{patient_id}/visits` 列表。`GET /patients/{patient_id}/full` 患者+病例+就诊+QA 打包。`POST /patients/{patient_id}/qa` 新增病例问答；`GET /patients/{patient_id}/qa/search` 查询病例问答（Query 可带 `query`、`limit`、`offset`）。

记忆设置：`GET /patients/{patient_id}/memory/settings`、`PATCH /patients/{patient_id}/memory/settings` 读写对话偏好。

短期会话：`GET /patients/by-code/{patient_code}/memory/session/{session_id}` 查看消息；`DELETE /patients/by-code/{patient_code}/memory/session/{session_id}` 删会话。

长期抽取：`POST /patients/by-code/{patient_code}/memory/extract-business`（无 body）。`POST /patients/by-code/{patient_code}/memory/extract-dialogue`（JSON：`session_id`）。

长期检索：`POST /patients/by-code/{patient_code}/memory/vector`（JSON：`operation`=`search`|`reindex`；`search` 时必填 `query`，可选 `top_k`、`search_mode`）。`GET /patients/{patient_id}/memory/extracted`（Query：`key_events_limit`、`key_events_offset`）。

Agent（主应用挂载）：`POST /api/agent/query-multimodal`（`multipart/form-data`，主入口；字段含 `query`、可选 `patient_code`、`phone`、`session_id`、`tts_voice`（`none`/不传不播报）、`image_url`、`image_file`）。成功时 `tool_name` 可能为 `agent.greeting`、`visit.query_visits`、`case.query_cases`、`auth.verify_identity`、`identity.request_verification` 等。`GET /api/agent/session-image/{image_id}`；`GET /api/agent/audio/{audio_id}`。`POST /api/agent/react-plan`（JSON：`user_query`、可选 `context`、`max_steps`、`sc_runs`、`include_all_episodes`）。

静态测试页：`GET /test/` 下静态页（如 `patient_app.html`、`memory_chat_test.html` 等，见 `backend/static`）。

依赖安装：项目根目录 **`requirements.txt`**，`pip install -r requirements.txt`（含 `fastapi`、`uvicorn[standard]`、`python-multipart`、`pydantic`、`dashscope`）。

独立 MCP 进程（非主应用端口，需单独 `uvicorn backend.mcp_server:app`）：`GET /health`、`GET /mcp/tools`、`POST /mcp/invoke`、`POST /mcp/agent-call`、`POST /mcp/agent-react`。

> 面试表达：对外可分三块——患者与业务 CRUD、记忆与抽取与向量、Agent 与 Planner；前端或测试页主要盯 query-multimodal 与 memory/vector。

---

## 16. 面试最爱问的问题（汇总）

下面按主题归纳「高频 + 和本项目强相关」的问题；准备时优先能用自己的话讲清：数据流、为什么这样设计、边界与改进。

### 架构与选型

- 为什么用 FastAPI、为什么 SQLite、单体会不会成为瓶颈？（答：原型与部署简单；瓶颈在并发写与向量全表扫，上生产可拆读写、换向量库、加缓存。）
- 为什么向量不单独上 Milvus / Chroma？（答：一体化、运维轻；数据量上来再迁移。）
- MCP 和主 Agent 为什么要两套路径？（答：主路径低延迟、规则路由；MCP 演示 LLM 选工具与多步规划，成本与场景不同。）

### 记忆与 RAG

- 短期和长期记忆怎么划分？存在哪几张表？（答：`session_memory` vs 关键事件、画像、偏好、向量块、FTS。）
- 向量存在哪、怎么检索？（答：`embedding_json` + 应用层余弦；混合检索再加 FTS5 bm25 与 RRF。）
- 混合检索为什么用 RRF、不用线性加权？（答：两路分数尺度不同，RRF 不依赖归一化，实现简单。）
- 如何避免模型胡编？（答：结构化病历先检索；记忆显式拼进 prompt；抽取有置信度与来源字段。）

### LLM 与成本

- 哪些步骤调模型？（答：抽取、embedding、VL、TTS、Planner / QwenRouter。）
- 怎么控成本与延迟？（答：规则路由减少无效 LLM；批量 embedding；Planner 与主问答分离；可配环境变量与开关。）
- Prompt 怎么组织的？（答：系统角色 + 用户侧分块：问题、记忆、检索结论、结构化 compact。）

### Agent 与工具

- ReAct、CoT、自我一致性在本项目里分别对应什么？（答：react_planner 多步 + cot_steps；QwenRouter 单步多采样投票。）
- 工具调用的输入输出怎么约束？（答：ToolSpec + JSON；失败时 observation 回传。）
- 多模态图片为什么限制公网 URL？（答：兼容 VL 服务可达性与安全。）

### 工程与质量

- 如何保证数据一致性（会话 vs 抽取）？（答：短期写入绑在 query-multimodal 成功后；抽取读库内会话。）
- 日志与排错怎么做？（答：`LOG_LEVEL`、`LOG_HTTP`、memory_vector 等关键路径 logging。）
- 若上线你还改什么？（答：鉴权、审计、向量库与异步索引、限流、监控。）

### 项目深挖（必背三道）

1. 从用户发一句问到返回答，链路是什么？（对照第 10 节讲一遍。）  
   **精简答（只讲思路）：** 先收用户问题与身份线索（编号或手机等），做意图路由，解析并校验患者；若问的是病例或就诊，就先查结构化库并生成摘要结论。再把长期记忆融进来：拉画像、偏好、本会话历史，并对已抽取的长期内容做全文与向量两路检索、融合排序，拼成记忆与检索上下文。若有图再走视觉理解；需要则合成语音。若本轮带会话标识，只在推理成功之后把用户句与助手答写入短期会话；否则直接返回带记忆与检索依据的文本答。

2. 长期记忆从产生到被检索到，经过哪些表与函数？（抽取 → 关键事件 → FTS + 向量 → hybrid_search。）  
   **精简答（只讲思路）：** 业务或对话经模型抽取后，关键事件落库，并同步维护全文索引；画像单独合并更新。对关键事件文本做向量化，向量块与事件关联存放。用户再问时，一路按全文相关性、一路按向量相似度分别排序，再用 RRF 之类方法融合两路结果，和画像等一起拼进提示，不参与胡编。

3. 你个人负责或最熟的是哪一块？准备 1 分钟 STAR：背景、你做了什么、指标或结果。  
   **精简答（可改人称与数字，只讲思路）：** 情境：医疗向 Agent 演示，无登录下用编号或手机把病历和多轮对话串起来，记忆与检索要能讲清楚来源。任务：我最熟的是记忆与检索这条线——短期怎么落、长期怎么抽、双索引怎么接到主问答。行动：抽取结果进关键事件并维护全文侧；向量侧用 embedding 建块；主链路里把混合检索和记忆上下文拼进模板，短期写入绑在「本轮答出来」之后，避免脏写。结果：演示上能稳定展示病历检索加长期记忆加多轮会话；路径可解释，以后换独立向量库也好迁移。（STAR 约 1 分钟，按你真实分工替换。）

> 面试表达：主动说「我们这是原型，边界在某某；若生产会某某」——面试官往往听的是你的判断力，不是背接口列表。

回答具体接口时可对照第 15 节完整路径。

---

## 附录：关键文件速查

`main.py`：主应用、记忆路由、日志中间件。  
`agent_module.py`：多模态问答、问候短路、身份交叉校验、文本路由、记忆合并、`tts_voice` 与 TTS。  
`patient_db.py`：SQLite 表结构、CRUD、FTS、向量块、`get_patient_by_phone`、画像键规范化。  
`memory_vector.py`：Embedding、混合检索、索引。  
`memory_extract.py`：对话与业务抽取 LLM。  
`mcp_server.py`：MCP 工具箱、QwenRouter、独立 FastAPI。  
`react_planner.py`：ReAct 与自我一致性。  
`react_api.py`：`/api/agent/react-plan`。  
`session_media.py`：多模态 content 拼接。

---

文档与当前代码一致；涉及身份文案、问候规则、`tts_voice` 或 `requirements.txt` 变更时，请同步更新 `README.md` 与本文件。
