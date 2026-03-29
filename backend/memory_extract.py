import json
import os
from typing import Any, Dict, List
from urllib import error as urllib_error
from urllib import request as urllib_request

SYSTEM_PROMPT_DIALOGUE = (
  "你是医疗信息抽取助手。输入仅为「用户与助手多轮对话」。\n"
  "请抽取：\n"
  "1) key_events：用户口述的症状、用药、计划等；source 固定为 dialogue；"
  "未与线下病历核对时 confidence 不超过 0.75。\n"
  "2) user_profile：沟通偏好、关注点、自述健康信息；键名必须与 output_schema 一致（中文键名）。\n"
  "不得把对话中未出现的内容当作事实写入。仅输出 JSON，无 markdown。"
)

SYSTEM_PROMPT_BUSINESS = (
  "你是医疗信息抽取助手。输入仅为「系统内病例与就诊记录摘要」，视为权威业务数据。\n"
  "请抽取：\n"
  "1) key_events：每条对应重要诊疗节点（诊断、复诊计划、医嘱要点等）；source 固定为 emr；confidence 可较高。\n"
  "2) user_profile：从病程归纳出的长期关注维度（如慢病管理重点），无则留空对象；键名必须与 output_schema 一致（中文键名）。\n"
  "不得编造记录中不存在的诊断或日期。仅输出 JSON，无 markdown。"
)


def _normalize_events(events: Any, default_source: str) -> List[Dict[str, Any]]:
  if not isinstance(events, list):
    return []
  out: List[Dict[str, Any]] = []
  for ev in events:
    if not isinstance(ev, dict):
      continue
    out.append(
      {
        "title": ev.get("title") or "未命名事件",
        "summary": ev.get("summary") or "",
        "event_date": ev.get("event_date"),
        "source": ev.get("source") or default_source,
        "confidence": ev.get("confidence"),
        "raw": ev,
      }
    )
  return out


def _call_qwen(system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    return {"key_events": [], "user_profile": {}, "error": "QWEN_API_KEY is not set"}

  base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
  model = os.getenv("QWEN_EXTRACT_MODEL", "qwen-plus")

  payload = {
    "model": model,
    "messages": [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ],
    "temperature": 0.1,
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
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
  except urllib_error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="ignore")
    return {"key_events": [], "user_profile": {}, "error": f"qwen_http_error: {e.code} {detail}"}
  except Exception as e:
    return {"key_events": [], "user_profile": {}, "error": f"qwen_extract_error: {e}"}

  events = parsed.get("key_events") or []
  profile = parsed.get("user_profile") or {}
  if not isinstance(profile, dict):
    profile = {}
  return {"parsed_events": events, "user_profile": profile, "error": None}


def run_extract_from_dialogue(
  *,
  dialogue_messages: List[Dict[str, Any]],
  patient_basic: Dict[str, Any],
) -> Dict[str, Any]:
  user_payload = {
    "patient_basic": patient_basic,
    "dialogue_messages": dialogue_messages,
    "output_schema": {
      "key_events": [
        {
          "title": "string",
          "summary": "string",
          "event_date": "YYYY-MM-DD or null",
          "source": "dialogue",
          "confidence": 0.0,
        }
      ],
      "user_profile": {
        "沟通风格": "string optional",
        "健康关注点": ["string"],
        "自述症状": ["string"],
        "随访关注点": ["string"],
        "备注说明": "string optional",
      },
    },
  }
  res = _call_qwen(SYSTEM_PROMPT_DIALOGUE, user_payload)
  if res.get("error"):
    return {"key_events": [], "user_profile": {}, "error": res["error"]}
  events = _normalize_events(res.get("parsed_events"), "dialogue")
  return {"key_events": events, "user_profile": res.get("user_profile") or {}, "error": None}


def run_extract_from_business(
  *,
  cases_compact: List[Dict[str, Any]],
  visits_compact: List[Dict[str, Any]],
  patient_basic: Dict[str, Any],
) -> Dict[str, Any]:
  user_payload = {
    "patient_basic": patient_basic,
    "medical_cases_summary": cases_compact,
    "visit_records_summary": visits_compact,
    "output_schema": {
      "key_events": [
        {
          "title": "string",
          "summary": "string",
          "event_date": "YYYY-MM-DD or null",
          "source": "emr",
          "confidence": 0.0,
        }
      ],
      "user_profile": {
        "长期关注重点": ["string"],
        "随访节奏": "string optional",
        "备注说明": "string optional",
      },
    },
  }
  res = _call_qwen(SYSTEM_PROMPT_BUSINESS, user_payload)
  if res.get("error"):
    return {"key_events": [], "user_profile": {}, "error": res["error"]}
  events = _normalize_events(res.get("parsed_events"), "emr")
  return {"key_events": events, "user_profile": res.get("user_profile") or {}, "error": None}
