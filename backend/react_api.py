"""统一 Agent 规划 HTTP 入口：与 MCP `POST /mcp/agent-call` 同源（`unified_planner`）。"""
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .mcp_server import MCPToolbox
from .patient_db import PatientDatabase
from .unified_planner import run_unified_agent_query

router = APIRouter(tags=["Planner"])
_db = PatientDatabase()


class ReactPlanRequest(BaseModel):
  user_query: str = Field(..., min_length=1, description="用户自然语言目标")
  context: Dict[str, Any] = Field(
    default_factory=dict,
    description="如 patient_id、patient_code、phone 等，供工具解析",
  )
  max_steps: Optional[int] = Field(default=None, ge=1, le=12, description="complex 时单轨迹最大步数，默认读环境变量")
  sc_runs: Optional[int] = Field(default=None, ge=1, le=8, description="complex 时 ReAct 并行轨迹数")
  include_all_episodes: bool = Field(
    default=False,
    description="complex 且为 true 时返回全部并行轨迹（响应体较大，仅调试使用）",
  )


class ReactPlanResponse(BaseModel):
  ok: bool
  mode: Optional[Literal["simple", "complex"]] = None
  strategy: Optional[str] = None
  final_answer: Optional[str] = None
  tool_name: Optional[str] = None
  router: Optional[Any] = None
  tool_result: Optional[Any] = None
  gate: Optional[Dict[str, Any]] = None
  vote: Optional[Dict[str, Any]] = None
  winner_trace: List[Dict[str, Any]] = Field(default_factory=list)
  episodes: Optional[List[Any]] = None
  error: Optional[str] = None
  run_errors: Optional[List[str]] = None


@router.post(
  "/api/agent/react-plan",
  response_model=ReactPlanResponse,
  summary="统一 Agent 规划（唯一对外入口）",
  description=(
    "模型在 prompt 中判定 simple/complex：simple 为单次 CoT 规划 + 工具自洽投票后执行一次；"
    "complex 自动走 ReAct+CoT+多轨迹自我一致性。与 MCP `POST /mcp/agent-call` 一致。需 QWEN_API_KEY。"
  ),
)
def react_plan(req: ReactPlanRequest) -> ReactPlanResponse:
  toolbox = MCPToolbox(db=_db)
  out = run_unified_agent_query(
    user_query=req.user_query.strip(),
    context=req.context or {},
    toolbox=toolbox,
    max_steps_override=req.max_steps,
    sc_runs_override=req.sc_runs,
    include_react_episodes=req.include_all_episodes,
  )

  if not out.get("ok"):
    return ReactPlanResponse(
      ok=False,
      error=out.get("error", "unknown"),
      run_errors=out.get("run_errors"),
    )

  mode = out.get("mode")
  inner = out.get("result") or {}

  if mode == "simple":
    return ReactPlanResponse(
      ok=True,
      mode="simple",
      tool_name=out.get("tool_name"),
      router=inner.get("router"),
      tool_result=inner.get("tool_result"),
      gate=inner.get("gate") if isinstance(inner.get("gate"), dict) else None,
    )

  return ReactPlanResponse(
    ok=True,
    mode="complex",
    strategy=inner.get("strategy"),
    final_answer=inner.get("final_answer"),
    vote=inner.get("vote"),
    winner_trace=inner.get("winner_trace") or [],
    gate=inner.get("gate") if isinstance(inner.get("gate"), dict) else None,
    episodes=inner.get("episodes"),
    run_errors=inner.get("run_errors"),
  )
