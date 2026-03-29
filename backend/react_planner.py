"""
ReAct + CoT + 自我一致性：多轮「思考→行动→观察」循环；多轨迹投票选优。

环境变量（可选）：
- QWEN_REACT_MODEL：默认与 QWEN_MODEL / qwen-plus 一致
- QWEN_REACT_MAX_STEPS：单条轨迹最大步数，默认 5
- QWEN_REACT_SC_RUNS：自我一致性并行轨迹数，默认 3
- QWEN_REACT_TEMPERATURE：第 2 条及以后轨迹的温度，默认 0.35；首条 0.1
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

REACT_SYSTEM_PROMPT = """你是一个医疗信息系统场景下的推理与行动规划智能体。

必须采用 ReAct 范式：在每一轮先推理，再决定是否调用工具或结束。
必须采用 CoT（思维链）：用 cot_steps 列出分步推理（中文短句，2～6 条）。

输出要求（仅输出一个 JSON 对象，不要 markdown）：
{
  "thought": "本轮核心判断（一句）",
  "cot_steps": ["1) ...", "2) ..."],
  "action": "<工具名> 或 \"finish\"",
  "arguments": { ... 与工具约定一致；action 为 finish 时可为 {} },
  "confidence": 0.0 到 1.0 的小数,
  "final_answer": "仅当 action 为 finish 时必填：面向用户的完整中文回答"
}

规则：
- action 必须是给定工具列表中的 name，或者是字符串 finish。
- 若已有足够观察可回答用户，action 选 finish，并在 final_answer 中作答；不得编造未在 observation 中出现的事实。
- 若需继续查库，选择唯一最相关的工具并填好 arguments。
- 若上下文缺少 patient_id / patient_code 等，仍可选最可能工具并在 thought 中说明风险。
"""


def _env_int(name: str, default: int) -> int:
  raw = os.getenv(name)
  if raw is None or not str(raw).strip():
    return default
  try:
    return max(1, int(raw))
  except ValueError:
    return default


def _chat_json(
  *,
  system_prompt: str,
  user_payload: Dict[str, Any],
  temperature: float,
) -> Dict[str, Any]:
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    raise RuntimeError("QWEN_API_KEY is not set")
  base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
  model = os.getenv("QWEN_REACT_MODEL", os.getenv("QWEN_MODEL", "qwen-plus"))
  payload = {
    "model": model,
    "messages": [
      {"role": "system", "content": system_prompt},
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
    with urllib_request.urlopen(req, timeout=90) as resp:
      body = json.loads(resp.read().decode("utf-8"))
  except urllib_error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"react_http_error: {e.code} {detail}") from e
  content = body["choices"][0]["message"]["content"]
  return json.loads(content)


def _truncate_obs(obs: Any, max_len: int = 6000) -> Any:
  s = json.dumps(obs, ensure_ascii=False)
  if len(s) <= max_len:
    return obs
  return {"_truncated": True, "preview": s[:max_len] + "…"}


def _summarize_trace_for_prompt(prior: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  out: List[Dict[str, Any]] = []
  for p in prior:
    o: Dict[str, Any] = {
      "thought": p.get("thought"),
      "action": p.get("action"),
      "arguments": p.get("arguments"),
    }
    if "observation" in p:
      o["observation"] = _truncate_obs(p.get("observation"), 4000)
    out.append(o)
  return out


def _parse_step(raw: Dict[str, Any], allowed_tools: List[str]) -> Dict[str, Any]:
  action = raw.get("action")
  if action is None:
    raise ValueError("missing_action")
  action = str(action).strip()
  if action != "finish" and action not in allowed_tools:
    raise ValueError(f"invalid_action: {action}")
  args = raw.get("arguments")
  if not isinstance(args, dict):
    args = {}
  conf = raw.get("confidence", 0.0)
  try:
    conf = float(conf)
  except (TypeError, ValueError):
    conf = 0.0
  cot = raw.get("cot_steps")
  if not isinstance(cot, list):
    cot = []
  cot = [str(x) for x in cot if x is not None][:12]
  out = {
    "thought": str(raw.get("thought") or "").strip(),
    "cot_steps": cot,
    "action": action,
    "arguments": args,
    "confidence": conf,
    "final_answer": raw.get("final_answer"),
  }
  if action == "finish":
    fa = out.get("final_answer")
    if not fa or not str(fa).strip():
      out["final_answer"] = "（模型未给出 final_answer，请根据已执行步骤与观察自行总结。）"
  return out


def run_react_episode(
  *,
  user_query: str,
  context: Dict[str, Any],
  tool_summary: List[Dict[str, Any]],
  invoke: Callable[[str, Dict[str, Any]], Any],
  allowed_tools: List[str],
  max_steps: int,
  temperature: float,
  sc_run_index: int = 0,
) -> Dict[str, Any]:
  """单条 ReAct 轨迹：多轮 Thought→Action→Observation。"""
  trace: List[Dict[str, Any]] = []
  last_tool_result: Any = None
  final_answer: Optional[str] = None
  terminal = "max_steps"

  for step_idx in range(max_steps):
    user_payload = {
      "user_query": user_query,
      "context": context,
      "tools": tool_summary,
      "prior_trace": _summarize_trace_for_prompt(trace),
      "step_index": step_idx,
      "max_steps": max_steps,
      "self_consistency_run_index": sc_run_index,
    }
    raw = _chat_json(system_prompt=REACT_SYSTEM_PROMPT, user_payload=user_payload, temperature=temperature)
    step = _parse_step(raw, allowed_tools)

    if step["action"] == "finish":
      final_answer = str(step.get("final_answer") or "").strip() or None
      trace.append({**step, "observation": None})
      terminal = "finish"
      break

    try:
      obs = invoke(step["action"], step["arguments"])
    except Exception as e:
      obs = {"error": str(e), "ok": False}
    last_tool_result = obs
    trace.append({**step, "observation": _truncate_obs(obs)})

  if final_answer is None:
    if last_tool_result is not None:
      final_answer = (
        "【基于已执行工具结果的简要说明】\n"
        + json.dumps(last_tool_result, ensure_ascii=False)[:4000]
      )
    else:
      final_answer = "未能在限定步数内完成规划；请补充患者标识或简化问题后重试。"

  confs = [float(s.get("confidence") or 0.0) for s in trace]
  avg_conf = sum(confs) / len(confs) if confs else 0.0

  chain: List[str] = []
  for s in trace:
    a = s.get("action")
    if a and a != "finish":
      chain.append(str(a))

  return {
    "trace": trace,
    "final_answer": final_answer,
    "last_tool_result": last_tool_result,
    "terminal": terminal,
    "avg_confidence": avg_conf,
    "action_chain": chain,
    "fingerprint": ">".join(chain) if chain else "finish",
  }


def vote_episodes(episodes: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """自我一致性：按 action 链指纹多数票，平局取平均置信度最高。"""
  if not episodes:
    raise ValueError("no_episodes")
  fps = [e.get("fingerprint") or "" for e in episodes]
  ctr = Counter(fps)
  winner_fp, winner_votes = ctr.most_common(1)[0]
  group = [e for e in episodes if (e.get("fingerprint") or "") == winner_fp]
  best = max(group, key=lambda e: float(e.get("avg_confidence") or 0.0))
  meta = {
    "votes": dict(ctr),
    "winner_fingerprint": winner_fp,
    "winner_votes": winner_votes,
    "runs": len(episodes),
  }
  return best, meta


def run_react_with_self_consistency(
  *,
  user_query: str,
  context: Dict[str, Any],
  tool_summary: List[Dict[str, Any]],
  invoke: Callable[[str, Dict[str, Any]], Any],
  allowed_tools: List[str],
  max_steps_override: Optional[int] = None,
  sc_runs_override: Optional[int] = None,
) -> Dict[str, Any]:
  sc_runs = sc_runs_override if sc_runs_override is not None else _env_int(
    "QWEN_REACT_SC_RUNS", _env_int("QWEN_PLANNER_SC_RUNS", 3)
  )
  max_steps = max_steps_override if max_steps_override is not None else _env_int("QWEN_REACT_MAX_STEPS", 5)
  base_temp = float(os.getenv("QWEN_REACT_TEMP_BASE", "0.1") or 0.1)
  sc_temp = float(os.getenv("QWEN_REACT_TEMPERATURE", os.getenv("QWEN_PLANNER_SC_TEMPERATURE", "0.35")) or 0.35)

  episodes: List[Dict[str, Any]] = []
  errors: List[str] = []
  for i in range(sc_runs):
    temp = base_temp if i == 0 else sc_temp
    try:
      ep = run_react_episode(
        user_query=user_query,
        context=context,
        tool_summary=tool_summary,
        invoke=invoke,
        allowed_tools=allowed_tools,
        max_steps=max_steps,
        temperature=temp,
        sc_run_index=i,
      )
      episodes.append(ep)
    except Exception as e:
      errors.append(str(e))

  if not episodes:
    return {
      "ok": False,
      "strategy": "react_cot_self_consistency",
      "error": "; ".join(errors) if errors else "all_runs_failed",
      "episodes": [],
    }

  best, vote_meta = vote_episodes(episodes)
  return {
    "ok": True,
    "strategy": "react_cot_self_consistency",
    "winner": best,
    "vote": vote_meta,
    "episodes": episodes,
    "run_errors": errors or None,
  }
