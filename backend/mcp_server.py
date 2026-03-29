import json
import os
from dataclasses import dataclass
from collections import Counter
from typing import Any, Dict, List, Optional
from urllib import error, request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.patient_db import PatientDatabase


# Shared prompt templates for planner policy.
SYSTEM_PROMPT = (
  "You are a medical MCP planner using ReAct-style tool planning.\n"
  "First reason internally, then output a compact JSON plan only.\n"
  "Do not output markdown.\n"
  "Allowed tools are exactly from provided list.\n"
  "Return JSON schema:\n"
  "{"
  "\"analysis\": {\"intent\":\"...\", \"evidence\":[\"...\"], \"risk\":\"low|medium|high\"},"
  "\"action\": {\"tool_name\":\"...\", \"arguments\": {...}},"
  "\"reason\": \"short rationale\","
  "\"confidence\": 0.0"
  "}"
)

PLANNER_REQUIREMENTS = [
  "Prefer exact identifier arguments from context",
  "If identifiers are missing, keep arguments minimal",
  "Keep reason short and factual",
]


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


class QwenRouter:
  """
  Qwen routing wrapper.

  Uses OpenAI-compatible Chat Completions API:
  - base_url default: https://dashscope.aliyuncs.com/compatible-mode/v1
  - model default: qwen-plus
  - key from env: QWEN_API_KEY
  """

  def __init__(self) -> None:
    self.api_key = os.getenv("QWEN_API_KEY", "").strip()
    self.base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    self.model = os.getenv("QWEN_MODEL", "qwen-plus")
    self.sc_runs = max(1, int(os.getenv("QWEN_PLANNER_SC_RUNS", "3")))
    self.sc_temperature = float(os.getenv("QWEN_PLANNER_TEMPERATURE", "0.35"))

  @property
  def enabled(self) -> bool:
    return bool(self.api_key)

  def route(self, user_query: str, tool_specs: List[ToolSpec], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return:
      {"tool_name":"...", "arguments":{...}, "reason":"..."}
    """
    fallback = self._fallback_route(user_query, context)
    if not self.enabled:
      fallback["planner"] = {"mode": "fallback_only"}
      return fallback

    tool_summary = [
      {"name": t.name, "description": t.description, "input_schema": t.input_schema}
      for t in tool_specs
    ]
    candidates: List[Dict[str, Any]] = []
    for i in range(self.sc_runs):
      try:
        one = self._route_once(
          user_query=user_query,
          context=context,
          tool_summary=tool_summary,
          temperature=self.sc_temperature if i > 0 else 0.1,
        )
        candidates.append(one)
      except Exception:
        continue

    if not candidates:
      fallback["reason"] = "fallback_after_qwen_error"
      fallback["planner"] = {"mode": "fallback_no_candidates"}
      return fallback

    # Self-consistency: majority vote by tool_name, then pick the highest confidence in winner group.
    name_counter = Counter([c.get("tool_name", "") for c in candidates if c.get("tool_name")])
    if not name_counter:
      fallback["reason"] = "fallback_after_qwen_invalid_candidates"
      fallback["planner"] = {"mode": "fallback_invalid_candidates"}
      return fallback
    winner_tool, winner_votes = name_counter.most_common(1)[0]
    winner_group = [c for c in candidates if c.get("tool_name") == winner_tool]
    winner = max(winner_group, key=lambda c: float(c.get("confidence") or 0.0))

    if "arguments" not in winner or not isinstance(winner["arguments"], dict):
      winner["arguments"] = {}
    winner["planner"] = {
      "mode": "react_cot_self_consistency",
      "votes": dict(name_counter),
      "winner_votes": winner_votes,
      "runs": len(candidates),
    }
    return winner

  def _route_once(
    self,
    *,
    user_query: str,
    context: Dict[str, Any],
    tool_summary: List[Dict[str, Any]],
    temperature: float,
  ) -> Dict[str, Any]:
    # ReAct-like + CoT-constrained JSON plan (reasoning summary only).
    user_prompt = {
      "user_query": user_query,
      "context": context,
      "tools": tool_summary,
      "requirements": PLANNER_REQUIREMENTS,
    }
    payload = {
      "model": self.model,
      "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
      ],
      "temperature": temperature,
      "response_format": {"type": "json_object"},
    }
    req = request.Request(
      url=f"{self.base_url}/chat/completions",
      data=json.dumps(payload).encode("utf-8"),
      headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}",
      },
      method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
      body = resp.read().decode("utf-8")
    data = json.loads(body)
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    action = parsed.get("action") if isinstance(parsed, dict) else None
    if not isinstance(action, dict):
      raise ValueError("qwen_planner_missing_action")
    tool_name = action.get("tool_name")
    if not tool_name:
      raise ValueError("qwen_planner_missing_tool_name")
    arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
    reason = parsed.get("reason") or "model_selected_tool"
    confidence = parsed.get("confidence", 0.0)
    analysis = parsed.get("analysis", {})
    return {
      "tool_name": tool_name,
      "arguments": arguments,
      "reason": reason,
      "confidence": confidence,
      "analysis": analysis,
    }

  def _fallback_route(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    q = (user_query or "").lower()
    if "身份" in user_query or "认证" in user_query or "verify" in q or "auth" in q:
      return {"tool_name": "auth.verify_identity", "arguments": context, "reason": "keyword_auth"}
    if "病例" in user_query or "case" in q:
      return {"tool_name": "case.query_cases", "arguments": context, "reason": "keyword_case"}
    return {"tool_name": "visit.query_visits", "arguments": context, "reason": "default_visit"}


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
  version="0.1.0",
  description="模块化 MCP Server：身份验证、病例查询、就诊记录调取；支持 Qwen 路由调用。",
  openapi_tags=[
    {"name": "System", "description": "系统检查"},
    {"name": "MCP", "description": "MCP 工具列表与手动调用"},
    {"name": "Agent", "description": "Qwen 路由 + 工具自动调用"},
  ],
)

db = PatientDatabase()
toolbox = MCPToolbox(db=db)
router_model = QwenRouter()


@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
  return {"status": "ok", "qwen_enabled": router_model.enabled, "model": router_model.model}


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


@app.post("/mcp/agent-call", response_model=MCPResponse, tags=["Agent"])
def agent_call(req: MCPAgentRequest) -> MCPResponse:
  try:
    route = router_model.route(req.user_query, toolbox.list_specs(), req.context)
    tool_name = route["tool_name"]
    args = route.get("arguments", {})
    result = toolbox.invoke(tool_name, args)
    return MCPResponse(
      ok=True,
      tool_name=tool_name,
      result={
        "router": route,
        "tool_result": result,
      },
    )
  except Exception as e:
    return MCPResponse(ok=False, error=str(e))


@app.post("/mcp/agent-react", tags=["Agent"], summary="ReAct+CoT+自我一致性多轮规划")
def mcp_agent_react(req: MCPAgentRequest) -> Dict[str, Any]:
  """与主服务 `POST /api/agent/react-plan` 等价逻辑（独立 MCP 进程时使用）。"""
  from .react_planner import run_react_with_self_consistency

  specs = toolbox.list_specs()
  tool_summary = [
    {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in specs
  ]
  allowed = [t.name for t in specs]

  def invoke(name: str, arguments: Dict[str, Any]) -> Any:
    return toolbox.invoke(name, arguments)

  return run_react_with_self_consistency(
    user_query=req.user_query.strip(),
    context=req.context or {},
    tool_summary=tool_summary,
    invoke=invoke,
    allowed_tools=allowed,
  )


if __name__ == "__main__":
  # Run:
  #   D:\anaconda\envs\py311\python.exe -m uvicorn backend.mcp_server:app --reload --port 8100
  pass

