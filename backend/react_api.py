"""ReAct + CoT + 自我一致性：HTTP 入口（与主应用 Patient Agent 共用数据库与 MCP 工具箱）。"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .mcp_server import MCPToolbox
from .patient_db import PatientDatabase
from .react_planner import run_react_with_self_consistency

router = APIRouter(tags=["Planner"])
_db = PatientDatabase()


class ReactPlanRequest(BaseModel):
  user_query: str = Field(..., min_length=1, description="用户自然语言目标")
  context: Dict[str, Any] = Field(
    default_factory=dict,
    description="如 patient_id、patient_code、phone 等，供工具解析",
  )
  max_steps: Optional[int] = Field(default=None, ge=1, le=12, description="单轨迹最大步数，默认读环境变量")
  sc_runs: Optional[int] = Field(default=None, ge=1, le=8, description="自我一致性并行轨迹数")
  include_all_episodes: bool = Field(
    default=False,
    description="为 true 时返回全部并行轨迹（响应体较大，仅调试使用）",
  )


class ReactPlanResponse(BaseModel):
  ok: bool
  strategy: str = "react_cot_self_consistency"
  final_answer: Optional[str] = None
  vote: Optional[Dict[str, Any]] = None
  winner_trace: List[Dict[str, Any]] = Field(default_factory=list)
  episodes: Optional[List[Any]] = None
  error: Optional[str] = None
  run_errors: Optional[List[str]] = None


@router.post(
  "/api/agent/react-plan",
  response_model=ReactPlanResponse,
  summary="ReAct+CoT+自我一致性多轮规划",
  description=(
    "多轮 Thought→Action→Observation；每步含 cot_steps 思维链。"
    "并行多条轨迹后对工具调用链做多数表决，在胜组内取平均置信度最高者。"
    "需 QWEN_API_KEY；工具与 MCP `/mcp/agent-call` 一致。"
  ),
)
def react_plan(req: ReactPlanRequest) -> ReactPlanResponse:
  toolbox = MCPToolbox(db=_db)
  specs = toolbox.list_specs()
  tool_summary = [
    {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in specs
  ]
  allowed = [t.name for t in specs]

  def invoke(name: str, args: Dict[str, Any]) -> Any:
    return toolbox.invoke(name, args)

  out = run_react_with_self_consistency(
    user_query=req.user_query.strip(),
    context=req.context or {},
    tool_summary=tool_summary,
    invoke=invoke,
    allowed_tools=allowed,
    max_steps_override=req.max_steps,
    sc_runs_override=req.sc_runs,
  )

  if not out.get("ok"):
    return ReactPlanResponse(
      ok=False,
      error=out.get("error", "unknown"),
      run_errors=out.get("run_errors"),
    )

  w = out["winner"]
  return ReactPlanResponse(
    ok=True,
    final_answer=w.get("final_answer"),
    vote=out.get("vote"),
    winner_trace=w.get("trace") or [],
    episodes=out.get("episodes") if req.include_all_episodes else None,
    run_errors=out.get("run_errors"),
  )
