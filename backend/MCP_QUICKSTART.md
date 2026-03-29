# 模块化 MCP Server 快速使用

## 1. 启动服务

在 `d:\Cursor` 目录执行：

```powershell
& "D:\anaconda\envs\py311\python.exe" -m uvicorn backend.mcp_server:app --reload --host 127.0.0.1 --port 8100
```

打开 Swagger：

- `http://127.0.0.1:8100/docs`

---

## 2. 工具列表

- `auth.verify_identity`：身份验证
- `case.query_cases`：病例查询
- `visit.query_visits`：就诊记录调取

可在 `GET /mcp/tools` 查看完整 schema。

---

## 2.1 MCP Streamable HTTP（规范单端点，非 SSE / 非 stdio）

- **端点**：`POST http://127.0.0.1:8100/mcp`（请求体为 **JSON-RPC 2.0**）。
- **响应**：`Content-Type: application/json`（单条 JSON-RPC `result` 或 `error`）；**不使用** `text/event-stream`。
- **会话**：首次 `initialize` 的响应头含 **`MCP-Session-Id`**，后续 `tools/list`、`tools/call` 须在请求头带上该值。
- **GET `/mcp`**：本实现返回 **405**（不提供 SSE 监听流，与「仅 JSON 响应」一致）。

**PowerShell 示例（初始化 + 列工具 + 调工具）**：

```powershell
$h = @{ "Content-Type"="application/json"; "Accept"="application/json"; "MCP-Protocol-Version"="2025-11-25" }
$init = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}'
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8100/mcp" -Method POST -Headers $h -Body $init
$sid = $r.Headers["MCP-Session-Id"]
$h2 = @{ "Content-Type"="application/json"; "Accept"="application/json"; "MCP-Protocol-Version"="2025-11-25"; "MCP-Session-Id"=$sid }
$body = '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
Invoke-RestMethod -Uri "http://127.0.0.1:8100/mcp" -Method POST -Headers $h2 -Body $body
```

通知 `notifications/initialized`（无 `id` 字段）将返回 **202** 且无 JSON 正文。

---

## 3. 手动调用工具（推荐先测）

接口：`POST /mcp/invoke`

示例（病例查询）：

```json
{
  "tool_name": "case.query_cases",
  "arguments": {
    "patient_code": "P1001",
    "limit": 10,
    "offset": 0
  }
}
```

---

## 4. 统一 Agent（Qwen：难易判定 → 单次工具或 ReAct）

接口：`POST /mcp/agent-call`（唯一 Agent 入口；与主应用 `POST /api/agent/react-plan` 同源逻辑）

示例：

```json
{
  "user_query": "帮我查询 P1001 的病例信息",
  "context": {
    "patient_code": "P1001",
    "limit": 10
  }
}
```

---

## 5. 配置 Qwen Key（你提供后填）

PowerShell 临时设置：

```powershell
$env:QWEN_API_KEY="你的QwenKey"
```

可选项：

```powershell
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen-plus"
```

重启 `uvicorn` 后生效。

> 未设置 `QWEN_API_KEY` 时，系统会自动使用关键词 fallback 路由（仍可工作）。

