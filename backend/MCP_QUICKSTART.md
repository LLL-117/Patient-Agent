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

## 4. Agent 自动路由调用（Qwen）

接口：`POST /mcp/agent-call`

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

