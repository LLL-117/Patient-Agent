import base64
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .memory_vector import hybrid_search
from .patient_db import PatientDatabase, USER_PROFILE_KEY_EN_TO_ZH
from .session_media import media_markdown_suffix


router = APIRouter(tags=["Agent"])
db = PatientDatabase()
AUDIO_DIR = Path(__file__).resolve().parent / "audio_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SESSION_IMG_DIR = Path(__file__).resolve().parent / "session_image_cache"
SESSION_IMG_DIR.mkdir(parents=True, exist_ok=True)


def _persist_session_image_data_url(data_url: str) -> Optional[str]:
  """本地上传图落盘，返回路径 /api/agent/session-image/{id}（供短期记忆 content 内短链引用，不返回 base64）。"""
  if not data_url or not data_url.strip().startswith("data:image/"):
    return None
  try:
    header, b64data = data_url.split(",", 1)
    raw = base64.b64decode(b64data)
    ext = "jpg"
    hl = header.lower()
    if "png" in hl:
      ext = "png"
    elif "webp" in hl:
      ext = "webp"
    elif "gif" in hl:
      ext = "gif"
    image_id = uuid4().hex
    path = SESSION_IMG_DIR / f"{image_id}.{ext}"
    path.write_bytes(raw)
    return f"/api/agent/session-image/{image_id}"
  except Exception:
    return None


MM_SYSTEM_PROMPT = (
  "你是医疗问答助手。请结合图片和结构化病历回答，"
  "先结论后证据再建议，不能替代医生诊断。"
  "回答正文必须使用简体中文，不要用英文作为回答主体。"
)

MM_USER_TEXT_TEMPLATE = (
  "【用户问题】\n{user_query}\n\n"
  "【背景参考：长期记忆、同会话历史与相关关键事件（请在回答中适当结合；未出现的信息勿编造）】\n"
  "{memory_context}\n\n"
  "【病历检索结论（结构化病历事实）】\n"
  "{text_answer}\n\n"
  "结构化数据：{compact_data}\n"
  "请结合图片给出简明中文回答（结论、证据、建议结构）。"
)

def _sanitize_public_image_url(url: Optional[str]) -> Optional[str]:
  """Qwen VL 需公网可访问的 http(s) URL；本地/内网会导致 400。"""
  if not url or not str(url).strip():
    return None
  u = str(url).strip()
  if not u.startswith(("http://", "https://")):
    return None
  low = u.lower()
  if "localhost" in low or "127.0.0.1" in low or "[::1]" in low:
    return None
  if ".local" in low or "192.168." in low or "10.0.0." in low or "10.0.1." in low:
    return None
  if low.startswith("https://172.") or low.startswith("http://172."):
    return None
  return u


class AgentQueryResponse(BaseModel):
  tool_name: str
  answer: str
  data: Dict[str, Any]


class AgentMultimodalQueryResponse(BaseModel):
  tool_name: str
  answer: Optional[str] = Field(
    default=None,
    description="无图片时返回文本检索与路由结果；有图片时为 null（该文本仍会作为多模态生成的依据，不单独返回）",
  )
  multimodal_answer: Optional[str] = Field(
    default=None,
    description="有图片时的多模态回答；无图片时为 null",
  )
  retrieval_answer: Optional[str] = Field(
    default=None,
    description="有图片时：病历/检索路径的文本结论（既往）；与 multimodal_answer（结合图片的当前结论）对应",
  )
  multimodal_degraded: bool = Field(
    default=False,
    description="多模态模型不可用或失败时为 true，此时 multimodal_answer 内已含说明，不再与 retrieval 分段",
  )
  multimodal_enabled: bool
  audio_enabled: bool = False
  audio_format: Optional[str] = None
  audio_url: Optional[str] = None
  audio_error: Optional[str] = None
  tts_voice: Optional[str] = Field(
    default=None,
    description="本轮语音合成使用的音色标识（Qwen 系统音色或 CosyVoice voice 参数）",
  )
  data: Dict[str, Any]


class AgentUnifiedQueryRequest(BaseModel):
  query: str = Field(..., description="自然语言问题")
  patient_code: Optional[str] = Field(default=None, description="患者编号；可与 phone 二选一")
  phone: Optional[str] = Field(
    default=None,
    description="11 位手机号；暂无登录时建议每次请求都填",
  )
  session_id: Optional[str] = Field(default=None, description="可选：仅用于把本轮问答写入短期记忆库")
  image_url: Optional[str] = Field(default=None, description="可选：公网可访问图片 URL")


def _extract_patient_code(text: str) -> Optional[str]:
  m = re.search(r"P\d{3,}", text, flags=re.IGNORECASE)
  return m.group(0).upper() if m else None


def _extract_phone(text: str) -> Optional[str]:
  m = re.search(r"1\d{10}", text)
  return m.group(0) if m else None


# 身份校验失败（编号/手机号不存在、不一致或格式不对）时统一返回给前端的提示
_IDENTITY_VERIFICATION_FAILED = "你输入的编号或手机号有误。"


def _normalize_cn_mobile_digits(phone: Optional[str]) -> Optional[str]:
  """将输入规范为 11 位数字；空或仅空白返回 None；格式非法抛出 400。"""
  if phone is None:
    return None
  d = re.sub(r"\D", "", str(phone).strip())
  if not d:
    return None
  if len(d) != 11 or not d.startswith("1"):
    raise HTTPException(status_code=400, detail=_IDENTITY_VERIFICATION_FAILED)
  return d


def _db_phone_digits(patient_phone: Optional[str]) -> str:
  return re.sub(r"\D", "", str(patient_phone or "").strip())


def _resolve_patient_identity(
  patient_code: Optional[str],
  phone_raw: Optional[str],
) -> Optional[Dict[str, Any]]:
  """
  解析患者身份。仅编号或仅手机号时按库查询；二者同时提供时必须对应同一患者，否则 400/404。
  """
  code = (patient_code or "").strip().upper() or None
  phone_n = _normalize_cn_mobile_digits(phone_raw) if phone_raw else None
  has_c = bool(code)
  has_p = bool(phone_n)

  if not has_c and not has_p:
    return None

  if has_c and has_p:
    p_code = db.get_patient(patient_code=code)
    p_phone = db.get_patient_by_phone(phone_n) if phone_n else None
    if p_code and p_phone and p_code["patient_id"] == p_phone["patient_id"]:
      return p_code
    if p_code and p_phone and p_code["patient_id"] != p_phone["patient_id"]:
      raise HTTPException(status_code=400, detail=_IDENTITY_VERIFICATION_FAILED)
    if p_code and not p_phone:
      dbp = _db_phone_digits(p_code.get("phone"))
      if not dbp:
        raise HTTPException(status_code=400, detail=_IDENTITY_VERIFICATION_FAILED)
      if dbp != phone_n:
        raise HTTPException(status_code=400, detail=_IDENTITY_VERIFICATION_FAILED)
      return p_code
    if not p_code and p_phone:
      own = (p_phone.get("patient_code") or "").strip().upper()
      if own != code:
        raise HTTPException(status_code=400, detail=_IDENTITY_VERIFICATION_FAILED)
      return p_phone
    raise HTTPException(status_code=404, detail=_IDENTITY_VERIFICATION_FAILED)

  if has_c:
    p = db.get_patient(patient_code=code)
    if not p:
      raise HTTPException(status_code=404, detail=_IDENTITY_VERIFICATION_FAILED)
    return p

  assert phone_n is not None
  p = db.get_patient_by_phone(phone_n)
  if not p:
    raise HTTPException(status_code=404, detail=_IDENTITY_VERIFICATION_FAILED)
  return p


# 纯问候时返回，不查库、不拼病历记忆块（与产品侧欢迎话术一致）
_GREETING_REPLY = (
  "您好！我是医院患者智能辅助 Agent，很高兴为您服务。请问有什么我可以帮您的吗？"
  "如果您需要查询病例、就诊记录或进行身份验证，请告诉我您的需求哦~"
)


# 零宽/BOM 等会导致「你好」等短句匹配失败，从而误走就诊查询
_GREETING_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\ufe0e\ufe0f]")


def _normalize_text_for_greeting(text: str) -> str:
  t = (text or "").strip()
  t = _GREETING_INVISIBLE_RE.sub("", t)
  return unicodedata.normalize("NFC", t)


def _is_greeting_query(text: str) -> bool:
  """仅短句问候、无业务关键词、无患者标识时视为问候。"""
  raw = _normalize_text_for_greeting(text)
  if not raw or len(raw) > 36:
    return False
  if _extract_patient_code(raw) or _extract_phone(raw):
    return False
  biz = (
    "病例",
    "就诊",
    "患者",
    "查询",
    "记录",
    "身份",
    "验证",
    "编号",
    "手机",
    "检查",
    "报告",
    "药",
    "症状",
    "诊断",
    "医生",
    "挂号",
    "化验",
  )
  if any(k in raw for k in biz):
    return False
  t = re.sub(r"\s+", "", raw)
  t = re.sub(r"[！!。.?？~～，,、…]+", "", t)
  if not t or len(t) > 18:
    return False
  if re.fullmatch(
    r"(您好|你好|嗨|哈喽|在吗|在么|早上好|中午好|下午好|晚上好|早|在不在)(呀|啊|哦|呢|哈|喽|咯|哇|呐|咧)?",
    t,
    flags=re.IGNORECASE,
  ):
    return True
  if re.fullmatch(r"(hi|hello|hey|hiya)([!.\?？！!])*", t, flags=re.IGNORECASE):
    return True
  return False


def _route_query(text: str) -> str:
  t = text.lower()
  if any(k in text for k in ["身份", "认证", "验证"]) or "verify" in t or "auth" in t:
    return "auth.verify_identity"
  if "病例" in text or "case" in t:
    return "case.query_cases"
  if any(k in text for k in ["就诊", "门诊", "复诊", "记录"]) or "visit" in t:
    return "visit.query_visits"
  return "visit.query_visits"


def _infer_query_limit(text: str, default_limit: int = 20) -> int:
  t = (text or "").lower()
  if any(k in text for k in ["最近", "最新"]) or "recent" in t or "latest" in t or "last" in t:
    return 1
  return default_limit


# 抽取画像字段名 → 展示用中文（避免回答里出现 chronic_focus 等英文键）
_PROFILE_KEY_LABEL_ZH: Dict[str, str] = {
  "chronic_focus": "长期关注重点",
  "care_rhythm": "随访节奏",
  "notes": "备注说明",
  "health_focus": "健康关注点",
  "self_reported_symptoms": "自述症状",
  "follow_up_concerns": "随访关注点",
  "communication_style": "沟通风格",
  # 已是中文键时直接展示
  "长期关注重点": "长期关注重点",
  "随访节奏": "随访节奏",
  "备注说明": "备注说明",
  "健康关注点": "健康关注点",
  "自述症状": "自述症状",
  "随访关注点": "随访关注点",
  "沟通风格": "沟通风格",
}


def _normalize_profile_value_zh(v: Any) -> str:
  """展示层将常见医学英文缩写换为中文，减少回答里夹杂英文。"""
  s = str(v).strip()
  if not s:
    return s
  out = s
  out = out.replace("GERD", "胃食管反流病")
  # Hp 紧接中文时 \b 不适用，用前后非英文字母边界
  out = re.sub(r"(?<![A-Za-z])Hp(?![A-Za-z])", "幽门螺杆菌", out, flags=re.IGNORECASE)
  return out


def _format_profile_lines(profile: Dict[str, Any], max_items: int = 40) -> str:
  if not profile:
    return ""
  lines: List[str] = []
  for i, (k, v) in enumerate(profile.items()):
    if i >= max_items:
      break
    if v is None or v == "":
      continue
    label = _PROFILE_KEY_LABEL_ZH.get(k, k)
    if isinstance(v, list):

      def _disp_one(x: Any) -> str:
        if isinstance(x, dict):
          return _normalize_profile_value_zh(json.dumps(x, ensure_ascii=False))
        if isinstance(x, list):
          return json.dumps(x, ensure_ascii=False)
        return _normalize_profile_value_zh(x)

      v_disp = "、".join(_disp_one(x) for x in v)
    else:
      v_disp = _normalize_profile_value_zh(v)
    lines.append(f"- {label}：{v_disp}")
  return "\n".join(lines)


def _sanitize_profile_keys_in_text(text: str) -> str:
  """将模型复述或旧数据中的英文画像键替换为中文（与落库 normalize 一致）。"""
  if not text:
    return text
  for en, zh in USER_PROFILE_KEY_EN_TO_ZH.items():
    if en == zh:
      continue
    text = re.sub(rf"(?m)^\s*{re.escape(en)}\s*[：:]\s*", f"{zh}：", text)
    text = text.replace(f"{en}：", f"{zh}：")
    text = text.replace(f"{en}:", f"{zh}：")
  return text


def _format_session_messages(messages: List[Dict[str, Any]], max_chars_per_msg: int = 900) -> str:
  lines: List[str] = []
  for m in messages:
    role = (m.get("role") or "?").strip()
    content = (m.get("content") or "").strip().replace("\r\n", "\n")
    if len(content) > max_chars_per_msg:
      content = content[:max_chars_per_msg] + "…"
    if not content:
      continue
    label = {"user": "用户", "assistant": "助手", "system": "系统"}.get(role, role)
    lines.append(f"{label}：{content}")
  return "\n".join(lines)


def _build_memory_context_block(
  patient_id: str,
  user_query: str,
  session_id: Optional[str],
) -> str:
  """长期画像、对话偏好、同会话历史、向量检索关键事件。"""
  parts: List[str] = []

  try:
    prof = db.get_extracted_user_profile(patient_id=patient_id)
    p = prof.get("profile") or {}
    if isinstance(p, dict) and p:
      pl = _format_profile_lines(p)
      if pl:
        parts.append("【用户画像（抽取）】\n" + pl)
  except Exception:
    pass

  try:
    settings = db.get_memory_settings(patient_id=patient_id)
    prefs = settings.get("preferences") or {}
    if isinstance(prefs, dict) and prefs:
      frag = json.dumps(prefs, ensure_ascii=False)
      if len(frag) > 800:
        frag = frag[:800] + "…"
      parts.append("【对话偏好】\n" + frag)
  except Exception:
    pass

  sid = (session_id or "").strip()
  if sid:
    try:
      sess = db.get_session_memory(patient_id=patient_id, session_id=sid, newest_first=False)
      msgs = sess.get("messages") or []
      if msgs:
        tail = msgs[-24:]
        sl = _format_session_messages(tail)
        if sl:
          parts.append("【本会话近期对话（不含本轮）】\n" + sl)
    except Exception:
      pass

  try:
    hits, err = hybrid_search(db, patient_id, user_query, top_k=6)
    if not err and hits:
      lines: List[str] = []
      for h in hits:
        ct = (h.get("content_text") or "").strip()
        if not ct:
          continue
        if len(ct) > 600:
          ct = ct[:600] + "…"
        lines.append(f"- {ct}")
      if lines:
        parts.append("【与当前问题语义相关的关键事件】\n" + "\n".join(lines))
  except Exception:
    pass

  block = _sanitize_profile_keys_in_text("\n\n".join(parts).strip())
  if len(block) > 8000:
    block = block[:8000] + "\n…（已截断）"
  return block


def _merge_memory_into_response(
  *,
  patient_id: str,
  user_query: str,
  session_id: Optional[str],
  tool_name: str,
  core_answer: str,
  data: Dict[str, Any],
) -> AgentQueryResponse:
  memory_block = ""
  if tool_name in ("case.query_cases", "visit.query_visits"):
    memory_block = _build_memory_context_block(patient_id, user_query, session_id)
  final = core_answer
  if memory_block:
    final = (
      "【病历与记录检索】\n"
      + core_answer
      + "\n\n【长期记忆与上下文】\n"
      + memory_block
    )
  out_data: Dict[str, Any] = {
    **data,
    "memory_context": memory_block or None,
    "retrieval_answer_core": core_answer,
    "has_memory_context": bool(memory_block),
  }
  return AgentQueryResponse(tool_name=tool_name, answer=final, data=out_data)


def _visit_query_wants_doctor(user_query: str) -> bool:
  """用户是否在问接诊/主治/出诊医生等（非泛泛查就诊记录）。"""
  q = user_query or ""
  return any(
    k in q
    for k in ("医生", "医师", "大夫", "主治", "接诊", "看诊", "出诊", "坐诊", "哪位医师")
  )


def _build_detailed_summary(
  *,
  patient_code: Optional[str],
  domain: str,
  items: list[Dict[str, Any]],
  user_query: str = "",
) -> str:
  if not items:
    label = "病例记录" if domain == "case" else "就诊记录"
    return f"已找到患者 {patient_code}，但暂未查询到{label}。"

  top = items[0]
  count = len(items)
  if domain == "case":
    title = top.get("case_title") or "未命名病例"
    diagnosis = top.get("diagnosis") or "未记录"
    onset = top.get("onset_date") or "未记录"
    evidences = []
    for it in items[:3]:
      evidences.append(
        f"{it.get('onset_date') or '日期未记'}：{it.get('case_title') or '未命名'}，诊断 {it.get('diagnosis') or '未记录'}"
      )
    return (
      f"结论：患者 {patient_code} 当前已检索到 {count} 条病例，最新重点为“{title}”（诊断：{diagnosis}，起病时间：{onset}）。\n\n"
      f"依据：{'；'.join(evidences)}。\n\n"
      "建议：结合近两周症状变化和检查结果复诊，必要时补充实验室/影像检查，由临床医生最终判定。"
    )

  visit_date = top.get("visit_date") or "未记录"
  dept = top.get("department") or "未记录"
  diagnosis = top.get("diagnosis") or "未记录"
  complaint = top.get("chief_complaint") or "未记录"
  doctor = top.get("doctor") or "未记录"

  if _visit_query_wants_doctor(user_query):
    evidences = []
    for it in items[:3]:
      evidences.append(
        f"{it.get('visit_date') or '日期未记'} {it.get('department') or '科室未记'}：接诊医生 {it.get('doctor') or '未记录'}，"
        f"主诉 {it.get('chief_complaint') or '未记录'}，诊断 {it.get('diagnosis') or '未记录'}"
      )
    return (
      f"结论：患者 {patient_code} 最近一次就诊（{visit_date}，{dept}）的接诊医生为「{doctor}」。\n\n"
      f"依据：{'；'.join(evidences)}。\n\n"
      "建议：如对用药或复诊安排有疑问，请向该院科室或主治团队进一步确认；出现明显加重请及时线下就医。"
    )

  evidences = []
  for it in items[:3]:
    evidences.append(
      f"{it.get('visit_date') or '日期未记'} {it.get('department') or '科室未记'}：主诉 {it.get('chief_complaint') or '未记录'}，诊断 {it.get('diagnosis') or '未记录'}"
    )
  return (
    f"结论：患者 {patient_code} 共检索到 {count} 条就诊记录，最近一次为 {visit_date}（{dept}），主诉“{complaint}”，诊断“{diagnosis}”。\n\n"
    f"依据：{'；'.join(evidences)}。\n\n"
    "建议：优先按最新诊疗计划执行，并结合症状变化按时复诊；如出现明显加重或警示症状，请及时线下就医。"
  )


def _build_qwen_multimodal_answer(
  *,
  user_query: str,
  memory_context: str,
  text_answer: str,
  tool_name: str,
  tool_data: Dict[str, Any],
  image_url: Optional[str],
  image_base64: Optional[str],
) -> str:
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    raise RuntimeError("QWEN_API_KEY is not set")

  base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
  model = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")

  image_ref: Optional[str] = None
  if image_url and image_url.strip():
    image_ref = image_url.strip()
  elif image_base64 and image_base64.strip():
    b64 = image_base64.strip()
    if not b64.startswith("data:image/"):
      b64 = f"data:image/jpeg;base64,{b64}"
    image_ref = b64
  else:
    raise ValueError("image_url or image_base64 is required")

  compact_data = {
    "tool_name": tool_name,
    "patient_code": tool_data.get("patient_code"),
    "items_top3": (tool_data.get("items") or [])[:3],
  }
  mem = (memory_context or "").strip() or "（暂无：无画像/会话补充或未建立关键事件向量索引）"
  user_text = MM_USER_TEXT_TEMPLATE.format(
    user_query=user_query,
    memory_context=mem,
    text_answer=text_answer.strip() or "（无文本检索结论）",
    compact_data=json.dumps(compact_data, ensure_ascii=False),
  )

  payload = {
    "model": model,
    "messages": [
      {
        "role": "system",
        "content": MM_SYSTEM_PROMPT,
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": user_text},
          {"type": "image_url", "image_url": {"url": image_ref}},
        ],
      },
    ],
    "temperature": 0.2,
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
    with urllib_request.urlopen(req, timeout=45) as resp:
      result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]
  except urllib_error.HTTPError as e:
    detail = e.read().decode("utf-8", errors="ignore")
    raise RuntimeError(f"qwen_http_error: {e.code} {detail}")


def _tts_response_meta(resp: Any) -> Dict[str, Any]:
  """从 DashScope SDK 返回对象上尽量取出 status_code / code / message。"""
  out: Dict[str, Any] = {}
  if resp is None:
    return out
  if isinstance(resp, dict):
    for k in ("status_code", "code", "message", "output", "request_id"):
      if k in resp:
        out[k] = resp[k]
    return out
  for k in ("status_code", "code", "message", "request_id"):
    if hasattr(resp, k):
      try:
        out[k] = getattr(resp, k)
      except Exception:
        pass
  return out


def _friendly_tts_missing_audio_error(resp: Any) -> str:
  """无音频数据时生成前端可读说明（含配额类 403）。"""
  meta = _tts_response_meta(resp)
  status = meta.get("status_code")
  code = meta.get("code")
  msg_en = str(meta.get("message") or "").strip()
  code_s = str(code or "")

  if status == 403 and ("AllocationQuota" in code_s or "FreeTier" in code_s or "free tier" in msg_en.lower()):
    return (
      "语音服务额度不足：当前阿里云百炼「免费额度」已用尽或账号为「仅免费额度」模式。"
      "请在控制台关闭「仅使用免费额度」、开通按量计费并保证账户余额，或更换有效 API Key。"
    )
  if status == 403:
    tail = f"{code_s} {msg_en[:400]}".strip()
    return f"语音接口被拒绝（403）{(': ' + tail) if tail else ''}"

  if status and status != 200:
    return f"语音服务返回异常（HTTP {status}）{(': ' + msg_en[:400]) if msg_en else ''}"

  try:
    brief = json.dumps(meta, ensure_ascii=False)[:900]
  except Exception:
    brief = str(resp)[:900]
  return f"未收到语音数据。详情：{brief}"


def _sanitize_tts_voice_param(v: Optional[str]) -> Optional[str]:
  if v is None:
    return None
  s = str(v).strip()
  if not s:
    return None
  if len(s) > 64:
    s = s[:64]
  if not re.fullmatch(r"[A-Za-z0-9_-]+", s):
    return None
  return s


def _tts_voice_requests_synthesis(request_voice: Optional[str]) -> bool:
  """声音选择为「无」或未传时不合成语音（不再使用单独的 tts_enabled 开关）。"""
  if request_voice is None:
    return False
  s = str(request_voice).strip().lower()
  if not s or s in ("none", "__none__", "off", "silent"):
    return False
  return True


def _resolve_tts_voice(request_voice: Optional[str]) -> str:
  s = _sanitize_tts_voice_param(request_voice)
  if s:
    return s
  fb = (os.getenv("QWEN_TTS_VOICE") or "Cherry").strip()
  return _sanitize_tts_voice_param(fb) or "Cherry"


def _tts_voice_display_value(request_voice: Optional[str]) -> str:
  """响应里展示的音色：未播报时为 none。"""
  if not _tts_voice_requests_synthesis(request_voice):
    return "none"
  return _resolve_tts_voice(request_voice)


def _is_cosyvoice_voice(v: str) -> bool:
  """CosyVoice 预置音色（控制台「龙安洋」等）走 dashscope.audio.tts_v2；其余按 Qwen 系统音色处理。"""
  x = v.strip().lower()
  if x in ("longanyang", "longanhuan"):
    return True
  if x.startswith("long") and "_v3" in x:
    return True
  if x.startswith("longan") or x.startswith("longxiao") or x.startswith("longyu"):
    return True
  return False


def _build_cosyvoice_tts_audio(text: str, *, voice: str) -> Dict[str, Optional[str]]:
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    return {"audio_base64": None, "audio_format": None, "audio_error": "QWEN_API_KEY is not set"}
  try:
    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
  except Exception as e:
    return {
      "audio_base64": None,
      "audio_format": None,
      "audio_error": f"dashscope_cosyvoice_import_error: {e}",
    }

  dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip() or api_key
  model = os.getenv("COSYVOICE_TTS_MODEL", "cosyvoice-v3-flash").strip() or "cosyvoice-v3-flash"
  prev_key = getattr(dashscope, "api_key", None)
  synth: Any = None
  try:
    dashscope.api_key = dashscope_key
    synth = SpeechSynthesizer(
      model=model,
      voice=voice,
      format=AudioFormat.MP3_24000HZ_MONO_256KBPS,
    )
    raw = synth.call(text, timeout_millis=120000)
  except Exception as e:
    return {
      "audio_base64": None,
      "audio_format": None,
      "audio_error": f"cosyvoice_tts_error: {e}",
    }
  finally:
    dashscope.api_key = prev_key

  if not raw:
    lr = synth.get_response() if synth is not None else None
    return {
      "audio_base64": None,
      "audio_format": None,
      "audio_error": f"cosyvoice_empty_audio: {lr}",
    }

  try:
    b64 = base64.b64encode(bytes(raw)).decode("utf-8")
  except Exception as e:
    return {
      "audio_base64": None,
      "audio_format": None,
      "audio_error": f"cosyvoice_encode_error: {e}",
    }
  return {"audio_base64": b64, "audio_format": "mp3", "audio_error": None}


def _synthesize_tts(text: str, request_voice: Optional[str]) -> Tuple[Dict[str, Optional[str]], str]:
  v = _resolve_tts_voice(request_voice)
  if _is_cosyvoice_voice(v):
    return _build_cosyvoice_tts_audio(text, voice=v), v
  return _build_qwen_tts_audio(text, voice=v), v


def _build_qwen_tts_audio(text: str, *, voice: str) -> Dict[str, Optional[str]]:
  """
  Generate speech audio with Qwen TTS (SpeechSynthesizer / 系统音色如 Cherry)。
  """
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    return {"audio_base64": None, "audio_format": None, "audio_error": "QWEN_API_KEY is not set"}

  try:
    import dashscope
  except Exception as e:
    return {
      "audio_base64": None,
      "audio_format": None,
      "audio_error": f"dashscope_import_error: {e}",
    }

  model = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash")
  dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip() or api_key

  try:
    resp = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
      model=model,
      api_key=dashscope_key,
      text=text,
      voice=voice,
    )

    # SDK response may expose output in different shapes.
    # Try object attributes first, then dict-like fallback.
    output = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None)
    if output is None and isinstance(resp, dict):
      output = resp

    audio_b64 = None
    if isinstance(output, dict):
      audio_b64 = (
        output.get("audio_base64")
        or (output.get("audio") or {}).get("data")
      )
      audio_url = (
        (output.get("audio") or {}).get("url")
        or output.get("audio_url")
      )
      if not audio_b64 and audio_url:
        with urllib_request.urlopen(audio_url, timeout=30) as r:
          audio_b64 = base64.b64encode(r.read()).decode("utf-8")
    else:
      # Generic object access.
      audio = getattr(output, "audio", None) if output else None
      audio_b64 = getattr(audio, "data", None) if audio else None
      audio_url = getattr(audio, "url", None) if audio else None
      if not audio_b64 and audio_url:
        with urllib_request.urlopen(audio_url, timeout=30) as r:
          audio_b64 = base64.b64encode(r.read()).decode("utf-8")

    if not audio_b64:
      return {
        "audio_base64": None,
        "audio_format": None,
        "audio_error": _friendly_tts_missing_audio_error(resp),
      }

    return {
      "audio_base64": audio_b64,
      "audio_format": "mp3",
      "audio_error": None,
    }
  except Exception as e:
    return {
      "audio_base64": None,
      "audio_format": None,
      "audio_error": f"qwen_tts_sdk_error: {e}",
    }


def _tts_enabled() -> bool:
  val = os.getenv("TTS_ENABLED", "true").strip().lower()
  return val not in {"0", "false", "no", "off"}


def _run_text_query(
  *,
  query: str,
  patient_code_override: Optional[str] = None,
  phone_override: Optional[str] = None,
  session_id: Optional[str] = None,
  allow_greeting: bool = True,
) -> AgentQueryResponse:
  text = query.strip()
  if not text:
    raise HTTPException(status_code=400, detail="query is required")

  if allow_greeting and _is_greeting_query(text):
    return AgentQueryResponse(
      tool_name="agent.greeting",
      answer=_GREETING_REPLY,
      data={"greeting": True},
    )

  patient_code = patient_code_override or _extract_patient_code(text)
  phone_raw = phone_override or _extract_phone(text)
  tool_name = _route_query(text)
  query_limit = _infer_query_limit(text, default_limit=20)

  patient = _resolve_patient_identity(patient_code, phone_raw)
  phone_n = _normalize_cn_mobile_digits(phone_raw) if phone_raw else None

  has_identifier = bool(patient_code or phone_raw)
  if patient is None:
    if tool_name in ("visit.query_visits", "case.query_cases", "auth.verify_identity") and not has_identifier:
      return AgentQueryResponse(
        tool_name="identity.request_verification",
        answer=(
          "查询病例或就诊记录前，需要先确认您的身份。\n"
          "请任选一种方式提供信息（无登录场景下请在接口表单中填写，或在问题里写出）：\n"
          "1. 患者编号，例如 P1001\n"
          "2. 院内登记手机号（11 位）\n"
          "提供后我会继续为您查询；请勿在公开场合泄露他人信息。"
        ),
        data={
          "requires_verification": True,
          "pending_intent": tool_name,
          "submit_via": ["patient_code", "phone"],
        },
      )
    raise HTTPException(
      status_code=404,
      detail=_IDENTITY_VERIFICATION_FAILED if has_identifier else "patient_not_found_or_identifier_missing",
    )

  patient_id = patient["patient_id"]
  patient_code = patient.get("patient_code")

  if tool_name == "auth.verify_identity":
    verified = True
    reason = "verified"
    if phone_n:
      verified = _db_phone_digits(patient.get("phone")) == phone_n
      reason = "verified" if verified else "phone_not_match"
    answer = (
      f"身份验证通过：患者 {patient.get('name')}（{patient_code}）。"
      if verified
      else _IDENTITY_VERIFICATION_FAILED
    )
    return AgentQueryResponse(
      tool_name=tool_name,
      answer=answer,
      data={
        "verified": verified,
        "reason": reason,
        "patient_id": patient_id,
        "patient_code": patient_code,
        "name": patient.get("name"),
      },
    )

  if tool_name == "case.query_cases":
    items = db.list_medical_cases(patient_id=patient_id, limit=query_limit, offset=0)
    core = _build_detailed_summary(
      patient_code=patient_code, domain="case", items=items, user_query=text
    )
    return _merge_memory_into_response(
      patient_id=patient_id,
      user_query=text,
      session_id=session_id,
      tool_name=tool_name,
      core_answer=core,
      data={"patient_id": patient_id, "patient_code": patient_code, "items": items},
    )

  items = db.list_visit_records(patient_id=patient_id, limit=query_limit, offset=0)
  core = _build_detailed_summary(
    patient_code=patient_code, domain="visit", items=items, user_query=text
  )
  return _merge_memory_into_response(
    patient_id=patient_id,
    user_query=text,
    session_id=session_id,
    tool_name="visit.query_visits",
    core_answer=core,
    data={"patient_id": patient_id, "patient_code": patient_code, "items": items},
  )


def _run_multimodal(
  *,
  query: str,
  patient_code: Optional[str],
  phone: Optional[str],
  session_id: Optional[str],
  tts_voice: Optional[str],
  image_url: Optional[str],
  image_base64: Optional[str],
) -> AgentMultimodalQueryResponse:
  has_image = bool((image_url and image_url.strip()) or (image_base64 and image_base64.strip()))
  base = _run_text_query(
    query=query,
    patient_code_override=patient_code,
    phone_override=phone,
    session_id=session_id,
    allow_greeting=not has_image,
  )
  multimodal_enabled = bool(os.getenv("QWEN_API_KEY", "").strip())
  mm_answer: Optional[str] = None
  mm_degraded = False
  mem_ctx = ""
  if isinstance(base.data, dict):
    raw_m = base.data.get("memory_context")
    if isinstance(raw_m, str):
      mem_ctx = raw_m
  core_answer = base.answer
  if isinstance(base.data, dict) and base.data.get("retrieval_answer_core") is not None:
    core_answer = str(base.data.get("retrieval_answer_core") or "")
  if has_image:
    try:
      mm_answer = _build_qwen_multimodal_answer(
        user_query=query,
        memory_context=mem_ctx,
        text_answer=core_answer,
        tool_name=base.tool_name,
        tool_data=base.data,
        image_url=image_url,
        image_base64=image_base64,
      )
    except Exception:
      mm_degraded = True
      mm_answer = (
        base.answer
        + "\n\n（本轮图片未能解析，已仅根据病历检索结果回答；请使用公网图片 URL 或上传图片文件。）"
      )
  else:
    mm_answer = None

  # TTS：有图用多模态文案，无图用纯文本 answer（异常路径下 mm_answer 可能为空，回退 answer）
  final_text = (mm_answer if has_image else None) or base.answer
  want_tts = _tts_enabled() and _tts_voice_requests_synthesis(tts_voice)
  if want_tts:
    tts_res, tts_voice_used = _synthesize_tts(final_text, tts_voice)
    audio_base64 = tts_res.get("audio_base64")
    audio_format = tts_res.get("audio_format")
    audio_error = tts_res.get("audio_error")
  else:
    tts_voice_used = _tts_voice_display_value(tts_voice)
    audio_base64 = None
    audio_format = None
    if not _tts_enabled():
      audio_error = "tts_disabled_by_env"
    else:
      audio_error = "tts_not_requested"
  audio_url: Optional[str] = None
  if audio_base64 and audio_format:
    try:
      audio_id = uuid4().hex
      audio_path = AUDIO_DIR / f"{audio_id}.{audio_format}"
      audio_path.write_bytes(base64.b64decode(audio_base64))
      audio_url = f"/api/agent/audio/{audio_id}"
    except Exception as e:
      audio_error = f"audio_file_write_error: {e}"
  audio_enabled = bool(audio_url)

  out_data = dict(base.data) if isinstance(base.data, dict) else {}
  out_data["tts_voice"] = tts_voice_used

  return AgentMultimodalQueryResponse(
    tool_name=base.tool_name,
    answer=None if has_image else base.answer,
    multimodal_answer=mm_answer,
    retrieval_answer=base.answer if has_image else None,
    multimodal_degraded=mm_degraded,
    multimodal_enabled=multimodal_enabled,
    audio_enabled=audio_enabled,
    audio_format=audio_format,
    audio_url=audio_url,
    audio_error=audio_error,
    tts_voice=tts_voice_used,
    data=out_data,
  )


@router.post(
  "/api/agent/query-multimodal",
  response_model=AgentMultimodalQueryResponse,
  summary="多模态问答",
)
async def agent_query_multimodal(
  request: Request,
  query: str = Form(...),
  patient_code: Optional[str] = Form(
    default=None,
    description="患者编号（如 P1001）；可与 phone 二选一或同时提供",
  ),
  phone: Optional[str] = Form(
    default=None,
    description="11 位手机号。暂无登录时建议每次请求都填写，用于匹配患者身份",
  ),
  session_id: Optional[str] = Form(
    default=None,
    description="可选：会话 ID。传入则在本轮问答成功后写入短期记忆；同会话历史也会参与长期画像/关键事件与病历检索的整合（不含本轮用户句）",
  ),
  tts_voice: Optional[str] = Form(
    default=None,
    description="TTS 音色：传 none 或不传表示不播报；CosyVoice 如 longanyang；Qwen 系统音色如 Cherry",
  ),
  image_url: Optional[str] = Form(default=None),
  image_file: Optional[UploadFile] = File(default=None),
) -> AgentMultimodalQueryResponse:
  # Optional local file upload path.
  final_b64: Optional[str] = None
  image_url_in = _sanitize_public_image_url(image_url)

  if image_file is not None:
    content = await image_file.read()
    if not content:
      raise HTTPException(status_code=400, detail="empty_image_file")
    b64 = base64.b64encode(content).decode("utf-8")
    ext = (image_file.filename or "").lower()
    if ext.endswith(".png"):
      prefix = "data:image/png;base64,"
    elif ext.endswith(".webp"):
      prefix = "data:image/webp;base64,"
    elif ext.endswith(".gif"):
      prefix = "data:image/gif;base64,"
    else:
      prefix = "data:image/jpeg;base64,"
    final_b64 = prefix + b64

  resp = _run_multimodal(
    query=query,
    patient_code=patient_code,
    phone=phone,
    session_id=(session_id or "").strip() or None,
    tts_voice=tts_voice,
    image_url=image_url_in,
    image_base64=final_b64,
  )

  sid = (session_id or "").strip()
  if sid:
    pid = (resp.data or {}).get("patient_id")
    if pid:
      try:
        origin = str(request.base_url).rstrip("/")
        img_ref: Optional[str] = None
        if final_b64:
          img_ref = _persist_session_image_data_url(final_b64)

        user_line = query.strip()
        user_extras: Optional[Dict[str, Any]] = None
        if final_b64 or (image_url_in and image_url_in.strip()):
          user_line = f"{user_line}\n[本轮含图片输入]"
          user_extras = {"modalities": ["image"]}
          if image_url_in and image_url_in.strip():
            user_extras["image_url"] = image_url_in.strip()
          if img_ref:
            user_extras["image_ref"] = img_ref
          user_line += media_markdown_suffix(
            origin,
            image_url=user_extras.get("image_url"),
            image_ref=img_ref,
            audio_url=None,
          )
        db.append_session_message(pid, sid, "user", user_line, extras=user_extras)

        reply = (resp.multimodal_answer or resp.answer or "").strip()
        if reply:
          # 有图且 VL 成功：短期记忆正文分两节——既往检索 vs 结合图片的当前结论
          if (
            resp.retrieval_answer
            and resp.multimodal_answer
            and not resp.multimodal_degraded
          ):
            hdr = "【上下文与病历检索】"
            try:
              if not (resp.data or {}).get("has_memory_context"):
                hdr = "【病历检索结论】"
            except Exception:
              pass
            reply = (
              hdr + "\n"
              + resp.retrieval_answer.strip()
              + "\n\n【结合图片的结论】\n"
              + resp.multimodal_answer.strip()
            )
          # multimodal：含 VL；text_retrieval：仅文本检索模板
          asst_extras: Dict[str, Any] = {
            "answer_type": "multimodal" if resp.multimodal_answer else "text_retrieval",
          }
          if (
            resp.retrieval_answer
            and resp.multimodal_answer
            and not resp.multimodal_degraded
          ):
            asst_extras["answer_layout"] = "retrieval_plus_vision"
          mods: list[str] = []
          if resp.audio_url:
            mods.append("audio")
          if resp.multimodal_answer:
            mods.append("vision")
          if mods:
            asst_extras["modalities"] = mods
          if resp.audio_url:
            asst_extras["audio_url"] = resp.audio_url
            if resp.audio_format:
              asst_extras["audio_format"] = resp.audio_format
          reply += media_markdown_suffix(origin, audio_url=resp.audio_url)
          db.append_session_message(pid, sid, "assistant", reply, extras=asst_extras)
      except Exception:
        pass

  return resp


@router.get(
  "/api/agent/session-image/{image_id}",
  summary="短期记忆引用的本地上传图片（content 中为短链，非 base64）",
)
def get_agent_session_image(image_id: str) -> FileResponse:
  if not re.fullmatch(r"[0-9a-fA-F]{32}", image_id or ""):
    raise HTTPException(status_code=400, detail="invalid_image_id")
  for ext, media in (
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("png", "image/png"),
    ("webp", "image/webp"),
    ("gif", "image/gif"),
  ):
    p = SESSION_IMG_DIR / f"{image_id}.{ext}"
    if p.exists():
      return FileResponse(path=str(p), media_type=media, filename=p.name)
  raise HTTPException(status_code=404, detail="session_image_not_found")


@router.get(
  "/api/agent/audio/{audio_id}",
  summary="播放语音播报",
)
def get_agent_audio(audio_id: str) -> FileResponse:
  if not re.fullmatch(r"[0-9a-fA-F]{32}", audio_id or ""):
    raise HTTPException(status_code=400, detail="invalid_audio_id")
  for ext in ("mp3", "wav", "pcm"):
    audio_path = AUDIO_DIR / f"{audio_id}.{ext}"
    if audio_path.exists():
      media_type = "audio/mpeg" if ext == "mp3" else "application/octet-stream"
      return FileResponse(path=str(audio_path), media_type=media_type, filename=audio_path.name)
  raise HTTPException(status_code=404, detail="audio_not_found")

