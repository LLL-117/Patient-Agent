import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .agent_module import router as agent_router
from .react_api import router as react_router
from .memory_extract import run_extract_from_business, run_extract_from_dialogue
from .memory_vector import (
  fts_search_events,
  hybrid_search,
  index_key_events,
  reindex_all_key_events,
  vector_search,
)
from .patient_db import PatientDatabase


def _configure_logging() -> None:
  level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
  level = getattr(logging, level_name, logging.INFO)
  if not logging.root.handlers:
    logging.basicConfig(
      level=level,
      format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
      datefmt="%Y-%m-%dT%H:%M:%S",
    )
  else:
    logging.root.setLevel(level)
  logging.getLogger("uvicorn.access").setLevel(level)
  logging.getLogger("uvicorn.error").setLevel(level)


_configure_logging()
_log = logging.getLogger("http")
_log_app = logging.getLogger(__name__)


class _RequestLoggingMiddleware(BaseHTTPMiddleware):
  """HTTP 请求耗时与状态码；默认关闭，设 LOG_HTTP=1 开启。"""

  _SKIP_PREFIXES = ("/test/",)

  async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
    if (os.getenv("LOG_HTTP") or "0").strip().lower() not in {"1", "true", "yes"}:
      return await call_next(request)
    path = request.url.path
    if path == "/health" or path.startswith(self._SKIP_PREFIXES):
      return await call_next(request)
    start = time.perf_counter()
    try:
      response = await call_next(request)
    except Exception:
      ms = (time.perf_counter() - start) * 1000
      _log.exception("%s %s failed after %.1fms", request.method, path, ms)
      raise
    ms = (time.perf_counter() - start) * 1000
    _log.info("%s %s -> %s %.1fms", request.method, path, response.status_code, ms)
    return response


app = FastAPI(
  title="Patient Agent API",
  version="0.1.0",
  description="患者身份信息、病例与就诊记录的存储和读取接口。",
  openapi_tags=[
    {"name": "Agent", "description": "自然语言问答查询（身份验证/病例/就诊记录）"},
    {
      "name": "Planner",
      "description": "ReAct + CoT + 自我一致性多轮推理与工具规划（/api/agent/react-plan）",
    },
    {"name": "System", "description": "系统状态检查"},
    {"name": "Patient", "description": "患者身份信息写入与读取"},
    {
      "name": "Memory",
      "description": "记忆相关：短期会话读写通过 Agent 多模态接口写入；对话/业务抽取与长期向量操作用 patient_code；记忆设置与 extracted 查询仍可用 patient_id。",
    },
    {"name": "Case", "description": "病例信息写入与读取"},
    {"name": "Visit", "description": "就诊记录写入与读取"},
    {"name": "Aggregate", "description": "患者综合信息查询"},
  ],
)

app.add_middleware(_RequestLoggingMiddleware)


def _http_access_log_enabled() -> bool:
  return (os.getenv("LOG_HTTP") or "0").strip().lower() in {"1", "true", "yes"}


db = PatientDatabase()


@app.on_event("startup")
async def _log_startup() -> None:
  raw_http = os.getenv("LOG_HTTP")
  _log_app.info(
    "Patient Agent API startup | LOG_LEVEL=%s | HTTP access log=%s (env LOG_HTTP=%s; 未设置时默认关闭，需显式 LOG_HTTP=1)",
    os.getenv("LOG_LEVEL") or "INFO",
    "on" if _http_access_log_enabled() else "off",
    repr(raw_http),
  )


def _patient_id_from_code_or_404(patient_code: str) -> str:
  patient = db.get_patient(patient_code=(patient_code or "").strip())
  if not patient:
    raise HTTPException(status_code=404, detail="patient_not_found")
  return str(patient["patient_id"])


# Serve static test pages.
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/test", StaticFiles(directory=str(static_dir)), name="test")


class UpsertPatientRequest(BaseModel):
  patient_id: Optional[str] = Field(default=None, description="Optional: use when patient_id is already known")
  patient_code: Optional[str] = Field(default=None, description="Recommended external identifier (unique)")
  name: Optional[str] = None
  gender: Optional[str] = None
  birth_date: Optional[str] = None
  phone: Optional[str] = None
  id_number: Optional[str] = None
  address: Optional[str] = None
  raw_json: Optional[Dict[str, Any]] = None


class UpsertPatientResponse(BaseModel):
  patient_id: str


class GetPatientQueryResponse(BaseModel):
  patient: Dict[str, Any]


class AddMedicalCaseRequest(BaseModel):
  case_id: Optional[str] = None
  case_title: Optional[str] = None
  diagnosis: Optional[str] = None
  description: Optional[str] = None
  onset_date: Optional[str] = None
  raw_json: Optional[Dict[str, Any]] = None


class AddMedicalCaseResponse(BaseModel):
  case_id: str


class ListMedicalCasesResponse(BaseModel):
  items: List[Dict[str, Any]]


class AddVisitRecordRequest(BaseModel):
  visit_id: Optional[str] = None
  visit_date: Optional[str] = None
  department: Optional[str] = None
  chief_complaint: Optional[str] = None
  diagnosis: Optional[str] = None
  doctor: Optional[str] = None
  notes: Optional[str] = None
  source: Optional[str] = None
  raw_json: Optional[Dict[str, Any]] = None


class AddVisitRecordResponse(BaseModel):
  visit_id: str


class ListVisitRecordsResponse(BaseModel):
  items: List[Dict[str, Any]]


class PatientFullResponse(BaseModel):
  patient: Dict[str, Any]
  medical_cases: List[Dict[str, Any]]
  visit_records: List[Dict[str, Any]]
  case_qa: List[Dict[str, Any]]


class AddCaseQARequest(BaseModel):
  case_id: Optional[str] = None
  query: str
  answer: str
  source: Optional[str] = None
  tags: Optional[List[str]] = None


class AddCaseQAResponse(BaseModel):
  qa_id: str


class SearchCaseQAResponse(BaseModel):
  items: List[Dict[str, Any]]


class MemorySettingsResponse(BaseModel):
  patient_id: str
  preferences: Dict[str, Any] = Field(
    default_factory=dict,
    description="用户自定义对话偏好，例如：reply_style（brief|detailed）、language、tone、focus_topics 等",
  )
  created_at: Optional[str] = None
  updated_at: Optional[str] = None


class MemorySettingsUpdateRequest(BaseModel):
  preferences: Dict[str, Any] = Field(..., description="偏好键值")
  replace: bool = Field(
    default=False,
    description="true：整体覆盖；false：与已有 preferences 浅层合并（顶层键覆盖）",
  )


class SessionMemoryResponse(BaseModel):
  session_id: str
  patient_id: str
  messages: List[Dict[str, Any]]
  created_at: Optional[str] = None
  updated_at: Optional[str] = None


class DialogueExtractRequest(BaseModel):
  session_id: str = Field(..., description="短期会话 ID（与 query-multimodal 等写入的 session 一致）")


class MemoryExtractResponse(BaseModel):
  patient_id: str
  key_event_ids: List[str]
  profile: Dict[str, Any]
  extraction_error: Optional[str] = None


class MemoryExtractedBundleResponse(BaseModel):
  """一次返回已抽取的关键事件与融合画像（替代分别 GET key-events / profile-extracted）。"""

  patient_id: str
  key_events: List[Dict[str, Any]]
  profile: Dict[str, Any]
  profile_created_at: Optional[str] = None
  profile_updated_at: Optional[str] = None


class MemoryEventApiItem(BaseModel):
  id: str
  patient_id: str
  event_type: str = Field(default="conversation_medical_hint", description="对话提炼多为 conversation_medical_hint")
  event_time: str
  title: Optional[str] = None
  summary: str
  source_type: str = Field(default="conversation")
  source_id: str = Field(..., description="如 session-003，标识来源会话")


class DialogueExtractResponse(BaseModel):
  """对话提炼完整结果：含落库用的 key_event_ids / profile，以及结构化 memory_events。"""

  patient_id: str
  key_event_ids: List[str] = Field(default_factory=list)
  profile: Dict[str, Any] = Field(default_factory=dict)
  extraction_error: Optional[str] = None
  event_count: int = 0
  profile_updated: bool = False
  memory_events: List[MemoryEventApiItem] = Field(default_factory=list)


class MemoryVectorHit(BaseModel):
  chunk_id: Optional[str] = None
  source_type: Optional[str] = None
  source_id: Optional[str] = None
  content_text: Optional[str] = None
  score: float = 0.0
  fusion_score: Optional[float] = Field(default=None, description="混合检索 RRF 分量（hybrid）")
  bm25: Optional[float] = Field(default=None, description="FTS bm25 原始分（fts）")
  retrieval: Optional[str] = Field(
    default=None,
    description="vector | fts | hybrid",
  )


class MemoryVectorOpsRequest(BaseModel):
  operation: Literal["search", "reindex"] = Field(
    ...,
    description="search：对关键事件做语义检索；reindex：为该患者重建长期向量索引",
  )
  query: Optional[str] = Field(default=None, description="operation=search 时必填：自然语言查询")
  top_k: int = Field(default=8, ge=1, le=50, description="仅 search 时生效")
  search_mode: Literal["hybrid", "vector", "fts"] = Field(
    default="hybrid",
    description="hybrid：FTS5(bm25)+稠密向量 RRF；vector：仅余弦；fts：仅全文",
  )


class MemoryVectorOpsResponse(BaseModel):
  patient_id: str
  operation: str
  query: Optional[str] = None
  results: Optional[List[MemoryVectorHit]] = None
  indexed_events: Optional[int] = None
  total_chunks: Optional[int] = None
  error: Optional[str] = None


def _rows_to_memory_events_api(
  rows: List[Dict[str, Any]],
  *,
  patient_id: str,
  session_id: str,
) -> List[MemoryEventApiItem]:
  out: List[MemoryEventApiItem] = []
  sid_tag = (session_id or "").strip() or "unknown"
  for r in rows:
    src = (r.get("source") or "").strip().lower()
    if src == "dialogue":
      etype = "conversation_medical_hint"
      stype = "conversation"
      src_id = f"session-{sid_tag}"
    else:
      etype = "medical_key_event"
      stype = src or "extracted"
      src_id = "emr"
    et = r.get("created_at") or r.get("updated_at") or r.get("event_date") or ""
    out.append(
      MemoryEventApiItem(
        id=str(r.get("event_id") or ""),
        patient_id=patient_id,
        event_type=etype,
        event_time=str(et),
        title=r.get("title"),
        summary=(r.get("summary") or "").strip(),
        source_type=stype,
        source_id=src_id,
      )
    )
  return out


def _compact_cases_for_extract(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  out: List[Dict[str, Any]] = []
  for it in items[:20]:
    out.append(
      {
        "case_title": it.get("case_title"),
        "diagnosis": it.get("diagnosis"),
        "onset_date": it.get("onset_date"),
        "description": (it.get("description") or "")[:200],
      }
    )
  return out


def _compact_visits_for_extract(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  out: List[Dict[str, Any]] = []
  for it in items[:20]:
    out.append(
      {
        "visit_date": it.get("visit_date"),
        "department": it.get("department"),
        "chief_complaint": it.get("chief_complaint"),
        "diagnosis": it.get("diagnosis"),
        "notes": (it.get("notes") or "")[:200],
      }
    )
  return out


def _patient_basic_dict(patient: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "patient_id": patient.get("patient_id"),
    "patient_code": patient.get("patient_code"),
    "name": patient.get("name"),
    "gender": patient.get("gender"),
    "birth_date": patient.get("birth_date"),
  }


def _persist_extraction(patient_id: str, result: Dict[str, Any]) -> MemoryExtractResponse:
  err = result.get("error")
  key_ids: List[str] = []
  if result.get("key_events"):
    try:
      key_ids = db.insert_key_events(patient_id=patient_id, events=result["key_events"])
    except ValueError:
      raise HTTPException(status_code=404, detail="patient_not_found")
    if key_ids:
      try:
        v_err = index_key_events(db, patient_id, key_ids)
        if v_err:
          logging.getLogger(__name__).warning("memory_vector_index: %s", v_err)
      except Exception as e:
        logging.getLogger(__name__).warning("memory_vector_index_failed: %s", e)
  prof = result.get("user_profile") or {}
  merged = db.merge_extracted_user_profile(patient_id=patient_id, profile=prof)
  return MemoryExtractResponse(
    patient_id=patient_id,
    key_event_ids=key_ids,
    profile=merged.get("profile") or {},
    extraction_error=err,
  )


@app.get("/health", tags=["System"], summary="健康检查")
def health() -> Dict[str, str]:
  return {"status": "ok"}


@app.post("/patients/upsert", response_model=UpsertPatientResponse, tags=["Patient"], summary="新增或更新患者信息")
def upsert_patient(req: UpsertPatientRequest) -> UpsertPatientResponse:
  try:
    patient_id = db.upsert_patient(req.model_dump(exclude_none=True))
    return UpsertPatientResponse(patient_id=patient_id)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"internal_error: {e}")


@app.get("/patients/{patient_id}", response_model=GetPatientQueryResponse, tags=["Patient"], summary="按 patient_id 查询患者")
def get_patient(patient_id: str) -> GetPatientQueryResponse:
  patient = db.get_patient(patient_id=patient_id)
  if not patient:
    raise HTTPException(status_code=404, detail="patient_not_found")
  return GetPatientQueryResponse(patient=patient)


@app.get("/patients/by-code/{patient_code}", response_model=GetPatientQueryResponse, tags=["Patient"], summary="按 patient_code 查询患者")
def get_patient_by_code(patient_code: str) -> GetPatientQueryResponse:
  patient = db.get_patient(patient_code=patient_code)
  if not patient:
    raise HTTPException(status_code=404, detail="patient_not_found")
  return GetPatientQueryResponse(patient=patient)


@app.post("/patients/{patient_id}/cases", response_model=AddMedicalCaseResponse, tags=["Case"], summary="新增病例")
def add_medical_case(patient_id: str, req: AddMedicalCaseRequest) -> AddMedicalCaseResponse:
  try:
    case_id = db.add_medical_case(patient_id=patient_id, case=req.model_dump(exclude_none=True))
    return AddMedicalCaseResponse(case_id=case_id)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"internal_error: {e}")


@app.get("/patients/{patient_id}/cases", response_model=ListMedicalCasesResponse, tags=["Case"], summary="查询病例列表")
def list_medical_cases(
  patient_id: str,
  limit: int = Query(default=50, ge=1, le=200),
  offset: int = Query(default=0, ge=0),
) -> ListMedicalCasesResponse:
  try:
    items = db.list_medical_cases(patient_id=patient_id, limit=limit, offset=offset)
    return ListMedicalCasesResponse(items=items)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))


@app.post("/patients/{patient_id}/visits", response_model=AddVisitRecordResponse, tags=["Visit"], summary="新增就诊记录")
def add_visit_record(patient_id: str, req: AddVisitRecordRequest) -> AddVisitRecordResponse:
  try:
    visit_id = db.add_visit_record(patient_id=patient_id, visit=req.model_dump(exclude_none=True))
    return AddVisitRecordResponse(visit_id=visit_id)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"internal_error: {e}")


@app.get("/patients/{patient_id}/visits", response_model=ListVisitRecordsResponse, tags=["Visit"], summary="查询就诊记录列表")
def list_visit_records(
  patient_id: str,
  limit: int = Query(default=50, ge=1, le=200),
  offset: int = Query(default=0, ge=0),
) -> ListVisitRecordsResponse:
  try:
    items = db.list_visit_records(patient_id=patient_id, limit=limit, offset=offset)
    return ListVisitRecordsResponse(items=items)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))


@app.get("/patients/{patient_id}/full", response_model=PatientFullResponse, tags=["Aggregate"], summary="查询患者全量信息")
def get_patient_full(patient_id: str) -> PatientFullResponse:
  full = db.get_patient_full(patient_id=patient_id)
  if not full:
    raise HTTPException(status_code=404, detail="patient_not_found")
  return PatientFullResponse(**full)


@app.get(
  "/patients/{patient_id}/memory/settings",
  response_model=MemorySettingsResponse,
  tags=["Memory"],
  summary="查询长期记忆对话偏好",
  description="若只知道 P1003 等编号，请先 GET /patients/by-code/{patient_code} 获取 patient_id。",
)
def get_memory_settings(patient_id: str) -> MemorySettingsResponse:
  try:
    data = db.get_memory_settings(patient_id=patient_id)
    return MemorySettingsResponse(**data)
  except ValueError as e:
    if str(e) == "patient_not_found":
      raise HTTPException(status_code=404, detail="patient_not_found")
    raise HTTPException(status_code=400, detail=str(e))


@app.patch(
  "/patients/{patient_id}/memory/settings",
  response_model=MemorySettingsResponse,
  tags=["Memory"],
  summary="更新长期记忆对话偏好",
  description="replace=true 为整体覆盖；replace=false 为浅层合并（与旧 PUT / PATCH 行为一致）。",
)
def update_memory_settings(patient_id: str, req: MemorySettingsUpdateRequest) -> MemorySettingsResponse:
  try:
    if req.replace:
      data = db.put_memory_settings(patient_id=patient_id, preferences=req.preferences)
    else:
      data = db.patch_memory_settings(patient_id=patient_id, preferences=req.preferences)
    return MemorySettingsResponse(**data)
  except ValueError as e:
    if str(e) == "patient_not_found":
      raise HTTPException(status_code=404, detail="patient_not_found")
    raise HTTPException(status_code=400, detail=str(e))


@app.get(
  "/patients/by-code/{patient_code}/memory/session/{session_id}",
  response_model=SessionMemoryResponse,
  tags=["Memory"],
  summary="查询短期会话",
  description=(
    "路径参数：patient_code（如 P1003）与 session_id。无需 patient_id。"
    "messages 按时间倒序（最近一轮在最前）。"
  ),
)
def get_session_memory_by_patient_code(
  patient_code: str,
  session_id: str,
) -> SessionMemoryResponse:
  pid = _patient_id_from_code_or_404(patient_code)
  try:
    data = db.get_session_memory(patient_id=pid, session_id=session_id, newest_first=True)
    return SessionMemoryResponse(**data)
  except ValueError as e:
    if str(e) == "patient_not_found":
      raise HTTPException(status_code=404, detail="patient_not_found")
    raise HTTPException(status_code=400, detail=str(e))


@app.delete(
  "/patients/by-code/{patient_code}/memory/session/{session_id}",
  tags=["Memory"],
  summary="删除短期会话缓存",
  description="路径参数为 patient_code 与 session_id，无需 patient_id。",
)
def delete_session_memory(patient_code: str, session_id: str) -> Dict[str, str]:
  pid = _patient_id_from_code_or_404(patient_code)
  try:
    db.delete_session_memory(patient_id=pid, session_id=session_id)
    return {"status": "ok"}
  except ValueError as e:
    if str(e) == "patient_not_found":
      raise HTTPException(status_code=404, detail="patient_not_found")
    raise HTTPException(status_code=400, detail=str(e))


@app.post(
  "/patients/by-code/{patient_code}/memory/extract-business",
  response_model=MemoryExtractResponse,
  tags=["Memory"],
  summary="从业务数据提炼长期记忆关键事件",
  description="基于病例与就诊记录摘要抽取关键事件（及病程相关画像维度），不读短期会话。需 QWEN_API_KEY。",
)
def memory_extract_from_business(patient_code: str) -> MemoryExtractResponse:
  pid = _patient_id_from_code_or_404(patient_code)
  patient = db.get_patient(patient_id=pid)
  if not patient:
    raise HTTPException(status_code=404, detail="patient_not_found")

  cases_compact = _compact_cases_for_extract(
    db.list_medical_cases(patient_id=pid, limit=20, offset=0)
  )
  visits_compact = _compact_visits_for_extract(
    db.list_visit_records(patient_id=pid, limit=20, offset=0)
  )
  if not cases_compact and not visits_compact:
    raise HTTPException(
      status_code=400,
      detail="no_business_data: add cases or visits before business extract",
    )

  result = run_extract_from_business(
    cases_compact=cases_compact,
    visits_compact=visits_compact,
    patient_basic=_patient_basic_dict(patient),
  )
  return _persist_extraction(pid, result)


@app.post(
  "/patients/by-code/{patient_code}/memory/extract-dialogue",
  response_model=DialogueExtractResponse,
  tags=["Memory"],
  summary="从短期对话提炼长期记忆画像与对话事件",
  description=(
    "仅基于指定 session 的多轮消息，不读病例/就诊库。Body 传 session_id。"
    "会话内容须先通过 POST /api/agent/query-multimodal（带 session_id）等多轮写入短期记忆。"
    "返回 key_event_ids、融合 profile，以及 event_count、profile_updated、memory_events。需 QWEN_API_KEY。"
  ),
)
def memory_extract_from_dialogue(
  patient_code: str,
  req: DialogueExtractRequest,
) -> DialogueExtractResponse:
  pid = _patient_id_from_code_or_404(patient_code)
  patient = db.get_patient(patient_id=pid)
  if not patient:
    raise HTTPException(status_code=404, detail="patient_not_found")

  sid = (req.session_id or "").strip()
  if not sid:
    raise HTTPException(status_code=400, detail="session_id is required")

  sess = db.get_session_memory(patient_id=pid, session_id=sid, newest_first=False)
  dialogue_messages = sess.get("messages") or []
  if not dialogue_messages:
    raise HTTPException(
      status_code=400,
      detail="no_dialogue_in_session: write dialogue via POST /api/agent/query-multimodal with session_id first",
    )

  profile_before = db.get_extracted_user_profile(patient_id=pid).get("profile") or {}

  result = run_extract_from_dialogue(
    dialogue_messages=dialogue_messages,
    patient_basic=_patient_basic_dict(patient),
  )
  persisted = _persist_extraction(pid, result)
  profile_after = db.get_extracted_user_profile(patient_id=pid).get("profile") or {}
  profile_updated = profile_after != profile_before

  rows = db.get_key_events_by_ids(pid, persisted.key_event_ids) if persisted.key_event_ids else []
  events_out = _rows_to_memory_events_api(rows, patient_id=pid, session_id=sid)

  return DialogueExtractResponse(
    patient_id=persisted.patient_id,
    key_event_ids=persisted.key_event_ids,
    profile=persisted.profile,
    extraction_error=persisted.extraction_error,
    event_count=len(events_out),
    profile_updated=profile_updated,
    memory_events=events_out,
  )


@app.post(
  "/patients/by-code/{patient_code}/memory/vector",
  response_model=MemoryVectorOpsResponse,
  tags=["Memory"],
  summary="长期向量：检索或重建索引",
  description=(
    "Body 设 operation："
    "`search` — 关键事件检索：默认 **hybrid**（SQLite FTS5 bm25 + 稠密向量 RRF）；"
    "可用 `search_mode` 改为 `vector` 或 `fts`；需 query，可选 top_k。"
    "`reindex` — 同步 FTS5 全文索引并为至多 500 条关键事件重建向量。"
    "向量检索需 QWEN_API_KEY；Embeddings 默认 text-embedding-v3。"
    "新关键事件在 extract 落库后会写入 FTS 并尝试建向量。"
  ),
)
def memory_vector_ops(
  patient_code: str,
  req: MemoryVectorOpsRequest,
) -> MemoryVectorOpsResponse:
  pid = _patient_id_from_code_or_404(patient_code)
  _log_app.info(
    "memory/vector patient_code=%s patient_id=%s operation=%s search_mode=%s",
    patient_code,
    pid,
    req.operation,
    req.search_mode if req.operation == "search" else "-",
  )
  if req.operation == "search":
    q = (req.query or "").strip()
    if not q:
      raise HTTPException(status_code=400, detail="query required when operation is search")
    try:
      if req.search_mode == "vector":
        results, verr = vector_search(db, pid, q, top_k=req.top_k)
      elif req.search_mode == "fts":
        results, verr = fts_search_events(db, pid, q, top_k=req.top_k)
      else:
        results, verr = hybrid_search(db, pid, q, top_k=req.top_k)
    except ValueError as e:
      if str(e) == "patient_not_found":
        raise HTTPException(status_code=404, detail="patient_not_found")
      raise HTTPException(status_code=400, detail=str(e))
    if verr:
      if "QWEN_API_KEY" in verr:
        raise HTTPException(status_code=503, detail=verr)
      if verr == "empty_query":
        raise HTTPException(status_code=400, detail=verr)
      raise HTTPException(status_code=502, detail=verr)
    return MemoryVectorOpsResponse(
      patient_id=pid,
      operation="search",
      query=q,
      results=[MemoryVectorHit(**r) for r in results],
    )

  try:
    n, err = reindex_all_key_events(db, pid, limit=500)
    total = db.count_memory_vector_chunks_for_patient(pid)
  except ValueError as e:
    if str(e) == "patient_not_found":
      raise HTTPException(status_code=404, detail="patient_not_found")
    raise HTTPException(status_code=400, detail=str(e))
  if err and "QWEN_API_KEY" in err:
    raise HTTPException(status_code=503, detail=err)
  if err:
    raise HTTPException(status_code=502, detail=err)
  _log_app.info(
    "memory/vector reindex done patient_id=%s indexed_events=%s total_chunks=%s",
    pid,
    n,
    total,
  )
  return MemoryVectorOpsResponse(
    patient_id=pid,
    operation="reindex",
    indexed_events=n,
    total_chunks=total,
    error=None,
  )


@app.get(
  "/patients/{patient_id}/memory/extracted",
  response_model=MemoryExtractedBundleResponse,
  tags=["Memory"],
  summary="查询已抽取的关键事件与融合画像",
  description="一次返回关键事件列表与用户画像，替代原先分别查询两个接口。",
)
def get_memory_extracted_bundle(
  patient_id: str,
  key_events_limit: int = Query(default=50, ge=1, le=200),
  key_events_offset: int = Query(default=0, ge=0),
) -> MemoryExtractedBundleResponse:
  try:
    events = db.list_key_events(
      patient_id=patient_id,
      limit=key_events_limit,
      offset=key_events_offset,
    )
    prof = db.get_extracted_user_profile(patient_id=patient_id)
    return MemoryExtractedBundleResponse(
      patient_id=patient_id,
      key_events=events,
      profile=prof.get("profile") or {},
      profile_created_at=prof.get("created_at"),
      profile_updated_at=prof.get("updated_at"),
    )
  except ValueError as e:
    if str(e) == "patient_not_found":
      raise HTTPException(status_code=404, detail="patient_not_found")
    raise HTTPException(status_code=400, detail=str(e))


@app.post("/patients/{patient_id}/qa", response_model=AddCaseQAResponse, tags=["Case"], summary="新增病例问答")
def add_case_qa(patient_id: str, req: AddCaseQARequest) -> AddCaseQAResponse:
  try:
    qa_id = db.add_case_qa(
      patient_id=patient_id,
      case_id=req.case_id,
      query=req.query,
      answer=req.answer,
      source=req.source,
      tags=req.tags,
    )
    return AddCaseQAResponse(qa_id=qa_id)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"internal_error: {e}")


@app.get("/patients/{patient_id}/qa/search", response_model=SearchCaseQAResponse, tags=["Case"], summary="查询病例问答")
def search_case_qa(
  patient_id: str,
  query: Optional[str] = Query(default=None),
  limit: int = Query(default=20, ge=1, le=200),
  offset: int = Query(default=0, ge=0),
) -> SearchCaseQAResponse:
  try:
    items = db.search_case_qa(patient_id=patient_id, query=query, limit=limit, offset=offset)
    return SearchCaseQAResponse(items=items)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))


app.include_router(agent_router)
app.include_router(react_router)
