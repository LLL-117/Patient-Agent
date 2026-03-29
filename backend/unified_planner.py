"""
统一 Agent 规划：由模型在 prompt 中判断难易（simple / complex）。
- simple：单次 CoT 式 JSON 规划 + 多采样对 tool_name 自洽投票，再 invoke 一次。
- complex：自动走 ReAct + CoT + 多轨迹自我一致性（run_react_with_self_consistency）。

环境变量与既有 Qwen Chat 配置及 react_planner 一致：QWEN_API_KEY、QWEN_MODEL、QWEN_BASE_URL、
QWEN_PLANNER_SC_RUNS、QWEN_PLANNER_TEMPERATURE；复杂路径另受 QWEN_REACT_* 影响。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from urllib import error as urllib_error
from urllib import request as urllib_request

from backend.react_planner import run_react_with_self_consistency

# -----------------------------------------------------------------------------
# Gate：难易 +（simple 时）单步 action
# -----------------------------------------------------------------------------

GATE_SYSTEM_PROMPT = """You are a medical MCP planner. You MUST classify each user request by difficulty, then output one JSON object only (no markdown).

Difficulty (意图与任务结构判断):
- **simple**: A single tool call from the provided list is enough to satisfy the user (one query, one verification, or one list fetch). Use chain-of-thought style reasoning only inside the JSON fields (analysis / reason), not as free text outside JSON.
- **complex**: The goal requires **multiple** tool invocations in sequence, or combining outputs from more than one tool, or clarifying multi-hop tasks (e.g. verify then query different resources). In that case you MUST set difficulty to complex and **action** to null.

Output JSON schema (exact keys):
{
  "difficulty": "simple" | "complex",
  "analysis": {
    "intent": "short natural language intent",
    "evidence": ["bullet", "..."],
    "risk": "low" | "medium" | "high",
    "why_difficulty": "one sentence: why simple or complex"
  },
  "action": null | { "tool_name": "<name from list>", "arguments": { } },
  "reason": "short rationale",
  "confidence": 0.0
}

Rules:
- If difficulty is **simple**, action MUST be a non-null object with a valid tool_name from the list and arguments matching that tool.
- If difficulty is **complex**, action MUST be null.
- Prefer exact identifiers from context when filling arguments.
"""


GATE_PLANNER_REQUIREMENTS = [
  "Classify difficulty before anything else.",
  "simple = exactly one tool call suffices; complex = multiple tools or multi-hop needed.",
  "If complex, action must be null.",
  "If simple, action must name exactly one tool from the list.",
]


def _env_int(name: str, default: int) -> int:
  raw = os.getenv(name)
  if raw is None or not str(raw).strip():
    return default
  try:
    return max(1, int(raw))
  except ValueError:
    return default


def qwen_runtime_info() -> Dict[str, Any]:
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  return {
    "enabled": bool(api_key),
    "model": os.getenv("QWEN_MODEL", "qwen-plus"),
  }


def _fallback_route(user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
  q = (user_query or "").lower()
  if "身份" in user_query or "认证" in user_query or "verify" in q or "auth" in q:
    return {"tool_name": "auth.verify_identity", "arguments": dict(context) if context else {}, "reason": "keyword_auth"}
  if "病例" in user_query or "case" in q:
    return {"tool_name": "case.query_cases", "arguments": dict(context) if context else {}, "reason": "keyword_case"}
  return {"tool_name": "visit.query_visits", "arguments": dict(context) if context else {}, "reason": "default_visit"}


def _chat_gate_json(
  *,
  user_query: str,
  context: Dict[str, Any],
  tool_summary: List[Dict[str, Any]],
  temperature: float,
) -> Dict[str, Any]:
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    raise RuntimeError("QWEN_API_KEY is not set")
  base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
  model = os.getenv("QWEN_MODEL", "qwen-plus")
  user_payload = {
    "user_query": user_query,
    "context": context,
    "tools": tool_summary,
    "requirements": GATE_PLANNER_REQUIREMENTS,
  }
  payload = {
    "model": model,
    "messages": [
      {"role": "system", "content": GATE_SYSTEM_PROMPT},
      {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ],
    "temperature": temperature,
    "response_format": {"type": "json_object"},
  }
  req = urllib_request.Request(
    url=f"{base_url}/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
    },
    method="POST",
  )
  try:
    with urllib_request.urlopen(req, timeout=60) as resp:
      body = json.loads(resp.read().decode("utf-8"))
  except urllib_error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"gate_http_error: {e.code} {detail}") from e
  content = body["choices"][0]["message"]["content"]
  return json.loads(content)


def _parse_gate_response(raw: Dict[str, Any], allowed_tools: List[str]) -> Dict[str, Any]:
  if not isinstance(raw, dict):
    raise ValueError("gate_invalid_root")
  diff = str(raw.get("difficulty") or "").strip().lower()
  if diff not in ("simple", "complex"):
    raise ValueError("gate_invalid_difficulty")
  analysis = raw.get("analysis") if isinstance(raw.get("analysis"), dict) else {}
  reason = raw.get("reason") or "gate"
  try:
    confidence = float(raw.get("confidence", 0.0))
  except (TypeError, ValueError):
    confidence = 0.0
  out: Dict[str, Any] = {
    "difficulty": diff,
    "analysis": analysis,
    "reason": reason,
    "confidence": confidence,
  }
  action = raw.get("action")
  if diff == "complex":
    if action is not None and action != {}:
      # tolerate model sending empty object
      pass
    out["tool_name"] = None
    out["arguments"] = {}
    return out
  # simple
  if not isinstance(action, dict):
    raise ValueError("gate_simple_missing_action")
  tool_name = action.get("tool_name")
  if not tool_name or str(tool_name) not in allowed_tools:
    raise ValueError("gate_simple_invalid_tool")
  args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
  out["tool_name"] = str(tool_name)
  out["arguments"] = args
  return out


def _gate_once(
  *,
  user_query: str,
  context: Dict[str, Any],
  tool_summary: List[Dict[str, Any]],
  allowed_tools: List[str],
  temperature: float,
) -> Dict[str, Any]:
  raw = _chat_gate_json(
    user_query=user_query,
    context=context,
    tool_summary=tool_summary,
    temperature=temperature,
  )
  return _parse_gate_response(raw, allowed_tools)


def run_unified_agent_query(
  *,
  user_query: str,
  context: Dict[str, Any],
  toolbox: Any,
  max_steps_override: Optional[int] = None,
  sc_runs_override: Optional[int] = None,
  include_react_episodes: bool = False,
) -> Dict[str, Any]:
  """
  单一入口逻辑：先 gate（难易），再 simple 单工具或 complex ReAct。

  返回 dict（供 MCP / HTTP 包装）:
  - ok: bool
  - mode: "simple" | "complex"
  - tool_name, result 等见各分支
  """
  text = (user_query or "").strip()
  if not text:
    return {"ok": False, "error": "empty_query"}

  specs = toolbox.list_specs()
  tool_summary = [
    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
    for t in specs
  ]
  allowed = [t.name for t in specs]

  def invoke(name: str, args: Dict[str, Any]) -> Any:
    return toolbox.invoke(name, args)

  api_key = os.getenv("QWEN_API_KEY", "").strip()
  sc_runs = sc_runs_override if sc_runs_override is not None else _env_int("QWEN_PLANNER_SC_RUNS", 3)
  sc_temp = float(os.getenv("QWEN_PLANNER_TEMPERATURE", "0.35") or 0.35)

  if not api_key:
    fb = _fallback_route(text, context or {})
    tool_name = fb["tool_name"]
    args = fb.get("arguments") or {}
    try:
      tool_result = invoke(tool_name, args)
    except Exception as e:
      return {"ok": False, "error": str(e), "mode": "simple", "planner": {"mode": "fallback_no_qwen"}}
    return {
      "ok": True,
      "mode": "simple",
      "tool_name": tool_name,
      "result": {
        "router": {**fb, "planner": {"mode": "fallback_only", "reason": "no_qwen_key"}},
        "tool_result": tool_result,
        "gate": None,
      },
    }

  candidates: List[Dict[str, Any]] = []
  for i in range(sc_runs):
    try:
      one = _gate_once(
        user_query=text,
        context=context or {},
        tool_summary=tool_summary,
        allowed_tools=allowed,
        temperature=sc_temp if i > 0 else 0.1,
      )
      candidates.append(one)
    except Exception:
      continue

  if not candidates:
    fb = _fallback_route(text, context or {})
    tool_name = fb["tool_name"]
    args = fb.get("arguments") or {}
    try:
      tool_result = invoke(tool_name, args)
    except Exception as e:
      return {"ok": False, "error": str(e), "mode": "simple", "planner": {"mode": "fallback_after_gate_error"}}
    return {
      "ok": True,
      "mode": "simple",
      "tool_name": tool_name,
      "result": {
        "router": {**fb, "planner": {"mode": "fallback_after_gate_error"}},
        "tool_result": tool_result,
        "gate": {"candidates_failed": sc_runs},
      },
    }

  complex_votes = sum(1 for c in candidates if c.get("difficulty") == "complex")
  simple_votes = sum(1 for c in candidates if c.get("difficulty") == "simple")
  use_complex = complex_votes > simple_votes

  gate_meta = {
    "difficulty_votes": {"simple": simple_votes, "complex": complex_votes},
    "runs": len(candidates),
    "decision": "complex" if use_complex else "simple",
  }

  if use_complex:
    react_out = run_react_with_self_consistency(
      user_query=text,
      context=context or {},
      tool_summary=tool_summary,
      invoke=invoke,
      allowed_tools=allowed,
      max_steps_override=max_steps_override,
      sc_runs_override=sc_runs_override,
    )
    if not react_out.get("ok"):
      return {
        "ok": False,
        "error": react_out.get("error", "react_failed"),
        "mode": "complex",
        "result": {"gate": gate_meta, "react": react_out},
        "run_errors": react_out.get("run_errors"),
      }
    w = react_out.get("winner") or {}
    complex_result: Dict[str, Any] = {
      "gate": gate_meta,
      "strategy": react_out.get("strategy"),
      "final_answer": w.get("final_answer"),
      "vote": react_out.get("vote"),
      "winner_trace": w.get("trace") or [],
      "run_errors": react_out.get("run_errors"),
    }
    if include_react_episodes:
      complex_result["episodes"] = react_out.get("episodes")
    return {
      "ok": True,
      "mode": "complex",
      "tool_name": None,
      "result": complex_result,
    }

  simple_only = [c for c in candidates if c.get("difficulty") == "simple" and c.get("tool_name")]
  if not simple_only:
    fb = _fallback_route(text, context or {})
    tool_name = fb["tool_name"]
    args = fb.get("arguments") or {}
    try:
      tool_result = invoke(tool_name, args)
    except Exception as e:
      return {"ok": False, "error": str(e), "mode": "simple", "result": {"gate": gate_meta}}
    return {
      "ok": True,
      "mode": "simple",
      "tool_name": tool_name,
      "result": {
        "router": {**fb, "planner": {"mode": "fallback_no_simple_candidate", "gate": gate_meta}},
        "tool_result": tool_result,
        "gate": gate_meta,
      },
    }

  name_counter = Counter([c.get("tool_name", "") for c in simple_only if c.get("tool_name")])
  winner_tool, winner_votes = name_counter.most_common(1)[0]
  winner_group = [c for c in simple_only if c.get("tool_name") == winner_tool]
  winner = max(winner_group, key=lambda c: float(c.get("confidence") or 0.0))
  tool_name = winner["tool_name"]
  args = winner.get("arguments") if isinstance(winner.get("arguments"), dict) else {}

  try:
    tool_result = invoke(tool_name, args)
  except Exception as e:
    return {"ok": False, "error": str(e), "mode": "simple", "result": {"gate": gate_meta, "router": winner}}

  router = {
    "tool_name": tool_name,
    "arguments": args,
    "reason": winner.get("reason"),
    "confidence": winner.get("confidence", 0.0),
    "analysis": winner.get("analysis"),
    "planner": {
      "mode": "cot_self_consistency",
      "votes": dict(name_counter),
      "winner_votes": winner_votes,
      "runs": len(candidates),
      "gate": gate_meta,
    },
  }
  return {
    "ok": True,
    "mode": "simple",
    "tool_name": tool_name,
    "result": {
      "router": router,
      "tool_result": tool_result,
      "gate": gate_meta,
    },
  }
