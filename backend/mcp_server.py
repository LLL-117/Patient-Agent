from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.responses import Response

from backend.mcp_streamable_http import MCPSessionStore, mcp_endpoint_delete, run_mcp_post
from backend.patient_db import PatientDatabase
from backend.unified_planner import qwen_runtime_info, run_unified_agent_query


# ---------------------------
# Data models
# ---------------------------
class MCPInvokeRequest(BaseModel):
  tool_name: str
  arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPAgentRequest(BaseModel):
  user_query: str
  context: Dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
  ok: bool
  tool_name: Optional[str] = None
  result: Any = None
  error: Optional[str] = None


@dataclass
class ToolSpec:
  name: str
  description: str
  input_schema: Dict[str, Any]


class MCPToolbox:
  def __init__(self, db: PatientDatabase):
    self.db = db
    self.tools: Dict[str, Any] = {
      "auth.verify_identity": self.verify_identity,
      "case.query_cases": self.query_cases,
      "visit.query_visits": self.query_visits,
    }

  def list_specs(self) -> List[ToolSpec]:
    return [
      ToolSpec(
        name="auth.verify_identity",
        description="身份验证：根据 patient_id 或 patient_code + 手机号后4位做内部验证",
        input_schema={
          "type": "object",
          "properties": {
            "patient_id": {"type": "string"},
            "patient_code": {"type": "string"},
            "phone_last4": {"type": "string"},
          },
        },
      ),
      ToolSpec(
        name="case.query_cases",
        description="病例查询：按 patient_id 或 patient_code 获取病例列表",
        input_schema={
          "type": "object",
          "properties": {
            "patient_id": {"type": "string"},
            "patient_code": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
            "offset": {"type": "integer", "default": 0},
          },
        },
      ),
      ToolSpec(
        name="visit.query_visits",
        description="就诊记录调取：按 patient_id 或 patient_code 获取就诊记录",
        input_schema={
          "type": "object",
          "properties": {
            "patient_id": {"type": "string"},
            "patient_code": {"type": "string"},
            "limit": {"type": "integer", "default": 20},
            "offset": {"type": "integer", "default": 0},
          },
        },
      ),
    ]

  def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
    fn = self.tools.get(tool_name)
    if not fn:
      raise ValueError(f"unknown_tool: {tool_name}")
    return fn(**arguments)

  def _resolve_patient(self, patient_id: Optional[str] = None, patient_code: Optional[str] = None) -> Dict[str, Any]:
    if patient_id:
      patient = self.db.get_patient(patient_id=patient_id)
    elif patient_code:
      patient = self.db.get_patient(patient_code=patient_code)
    else:
      raise ValueError("missing_patient_identifier")
    if not patient:
      raise ValueError("patient_not_found")
    return patient

  def verify_identity(
    self,
    patient_id: Optional[str] = None,
    patient_code: Optional[str] = None,
    phone_last4: Optional[str] = None,
    **kwargs: Any,
  ) -> Dict[str, Any]:
    _ = kwargs
    patient = self._resolve_patient(patient_id=patient_id, patient_code=patient_code)
    phone = patient.get("phone") or ""
    ok = True
    reason = "verified"
    if phone_last4:
      ok = phone.endswith(str(phone_last4))
      reason = "verified" if ok else "phone_last4_not_match"
    return {
      "verified": ok,
      "reason": reason,
      "patient_id": patient["patient_id"],
      "patient_code": patient.get("patient_code"),
      "name": patient.get("name"),
    }

  def query_cases(
    self,
    patient_id: Optional[str] = None,
    patient_code: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    **kwargs: Any,
  ) -> Dict[str, Any]:
    _ = kwargs
    patient = self._resolve_patient(patient_id=patient_id, patient_code=patient_code)
    items = self.db.list_medical_cases(patient_id=patient["patient_id"], limit=limit, offset=offset)
    return {
      "patient_id": patient["patient_id"],
      "patient_code": patient.get("patient_code"),
      "total_returned": len(items),
      "items": items,
    }

  def query_visits(
    self,
    patient_id: Optional[str] = None,
    patient_code: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    **kwargs: Any,
  ) -> Dict[str, Any]:
    _ = kwargs
    patient = self._resolve_patient(patient_id=patient_id, patient_code=patient_code)
    items = self.db.list_visit_records(patient_id=patient["patient_id"], limit=limit, offset=offset)
    return {
      "patient_id": patient["patient_id"],
      "patient_code": patient.get("patient_code"),
      "total_returned": len(items),
      "items": items,
    }


app = FastAPI(
  title="Modular MCP Server",
  version="0.2.0",
  description=(
    "模块化 MCP Server：身份验证、病例查询、就诊记录调取；支持 Qwen 路由调用。\n\n"
    "**MCP Streamable HTTP**：`POST /mcp` 为 JSON-RPC 2.0 单端点，响应使用 **application/json**（非 SSE、非 stdio）。"
    "`GET /mcp` 返回 405（不提供 SSE 监听流）。兼容方法：`initialize`、`notifications/initialized`、`tools/list`、`tools/call`。"
    "旧版 REST：`/mcp/tools`、`/mcp/invoke` 等仍保留。"
  ),
  openapi_tags=[
    {"name": "System", "description": "系统检查"},
    {"name": "MCP", "description": "Streamable HTTP `/mcp` + 旧版 REST"},
    {"name": "Agent", "description": "统一规划：难易判定后单次工具或 ReAct 多步（仅 POST /mcp/agent-call）"},
  ],
)

db = PatientDatabase()
toolbox = MCPToolbox(db=db)
_mcp_sessions = MCPSessionStore()


@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
  info = qwen_runtime_info()
  return {"status": "ok", "qwen_enabled": info["enabled"], "model": info["model"]}


@app.post("/mcp", tags=["MCP"], summary="MCP Streamable HTTP（JSON-RPC，application/json 响应）")
async def mcp_streamable_post(request: Request) -> Response:
  """规范 2025-11-25：单端点 POST；非 SSE。客户端须 Accept 含 application/json（及规范中的 event-stream 可一并列出）。"""
  return await run_mcp_post(request, toolbox, _mcp_sessions)


@app.get("/mcp", tags=["MCP"], summary="不提供 SSE 流（返回 405）")
def mcp_streamable_get() -> Response:
  return Response(status_code=405, headers={"Allow": "POST, DELETE"})


@app.delete("/mcp", tags=["MCP"], summary="终止 MCP 会话（MCP-Session-Id）")
def mcp_streamable_delete(request: Request) -> Response:
  return mcp_endpoint_delete(request, _mcp_sessions)


@app.get("/mcp/tools", tags=["MCP"])
def list_tools() -> Dict[str, Any]:
  return {"tools": [t.__dict__ for t in toolbox.list_specs()]}


@app.post("/mcp/invoke", response_model=MCPResponse, tags=["MCP"])
def invoke_tool(req: MCPInvokeRequest) -> MCPResponse:
  try:
    result = toolbox.invoke(req.tool_name, req.arguments)
    return MCPResponse(ok=True, tool_name=req.tool_name, result=result)
  except Exception as e:
    return MCPResponse(ok=False, tool_name=req.tool_name, error=str(e))


@app.post(
  "/mcp/agent-call",
  response_model=MCPResponse,
  tags=["Agent"],
  summary="统一 Agent（唯一对外入口）",
  description=(
    "模型在 prompt 中判定 simple/complex；simple 为单次 CoT 规划 + 工具自洽投票后 invoke 一次；"
    "complex 自动走 ReAct+CoT+多轨迹自我一致性。与主服务 POST /api/agent/react-plan 同源逻辑。"
  ),
)
def agent_call(req: MCPAgentRequest) -> MCPResponse:
  try:
    out = run_unified_agent_query(
      user_query=req.user_query,
      context=req.context or {},
      toolbox=toolbox,
      include_react_episodes=False,
    )
    if not out.get("ok"):
      return MCPResponse(ok=False, error=out.get("error", "unknown"))
    return MCPResponse(
      ok=True,
      tool_name=out.get("tool_name"),
      result={
        "mode": out.get("mode"),
        **(out.get("result") or {}),
      },
    )
  except Exception as e:
    return MCPResponse(ok=False, error=str(e))


if __name__ == "__main__":
  # Run:
  #   D:\anaconda\envs\py311\python.exe -m uvicorn backend.mcp_server:app --reload --port 8100
  pass

