"""长期记忆向量：DashScope Embeddings + SQLite 向量列；关键事件全文用 SQLite FTS5；查询支持混合检索（RRF）。"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from urllib import error as urllib_error
from urllib import request as urllib_request

from .patient_db import PatientDatabase

logger = logging.getLogger(__name__)

SOURCE_KEY_EVENT = "key_event"
_EMBED_BATCH = 16
_RRF_K = max(1, int(os.getenv("MEMORY_HYBRID_RRF_K", "60")))
_CANDIDATE_MULT = max(2, int(os.getenv("MEMORY_HYBRID_CANDIDATE_MULT", "2")))


def _l2_norm(v: List[float]) -> float:
  return math.sqrt(sum(x * x for x in v)) or 1.0


def cosine_similarity(a: List[float], b: List[float]) -> float:
  if len(a) != len(b) or not a:
    return 0.0
  dot = sum(x * y for x, y in zip(a, b))
  return dot / (_l2_norm(a) * _l2_norm(b))


def embed_texts(texts: List[str]) -> Tuple[List[List[float]], Optional[str]]:
  """调用兼容模式 /embeddings。返回 (向量列表, 错误信息)。"""
  if not texts:
    return [], None
  api_key = os.getenv("QWEN_API_KEY", "").strip()
  if not api_key:
    return [], "QWEN_API_KEY is not set"
  base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
  model = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

  out_vecs: List[Optional[List[float]]] = [None] * len(texts)
  err: Optional[str] = None

  for start in range(0, len(texts), _EMBED_BATCH):
    batch = texts[start : start + _EMBED_BATCH]
    payload = {"model": model, "input": batch}
    req = urllib_request.Request(
      url=f"{base_url}/embeddings",
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
      logger.warning("embed_http_error batch=%s %s", start, e.code)
      return [], f"embed_http_error: {e.code} {detail}"
    except Exception as e:
      logger.warning("embed_error batch=%s: %s", start, e)
      return [], f"embed_error: {e}"

    items = body.get("data") or []
    items.sort(key=lambda x: int(x.get("index", 0)))
    if len(items) != len(batch):
      return [], "embed_response_mismatch: data length"
    for i, item in enumerate(items):
      emb = item.get("embedding")
      if not isinstance(emb, list):
        return [], "embed_invalid_embedding"
      out_vecs[start + i] = [float(x) for x in emb]

  merged = [v for v in out_vecs if v is not None]
  if len(merged) != len(texts):
    return [], "embed_incomplete"
  dim = len(merged[0])
  for v in merged:
    if len(v) != dim:
      return [], "embed_dimension_mismatch"
  return merged, err


def _key_event_text(ev: Dict[str, Any]) -> str:
  title = (ev.get("title") or "").strip()
  summary = (ev.get("summary") or "").strip()
  if title and summary:
    return f"{title}\n{summary}"
  return title or summary


def index_key_events(db: PatientDatabase, patient_id: str, event_ids: List[str]) -> Optional[str]:
  """将关键事件写入向量表；失败时返回错误字符串。"""
  logger.info("index_key_events patient_id=%s count=%s", patient_id, len(event_ids))
  if not event_ids:
    return None
  rows = db.get_key_events_by_ids(patient_id=patient_id, event_ids=event_ids)
  texts: List[str] = []
  meta: List[Tuple[str, str]] = []
  for ev in rows:
    eid = str(ev.get("event_id") or "")
    t = _key_event_text(ev)
    if not t or not eid:
      continue
    texts.append(t)
    meta.append((eid, t))
  if not texts:
    return None
  vecs, err = embed_texts(texts)
  if err:
    return err
  for (eid, content), vec in zip(meta, vecs):
    try:
      db.upsert_memory_vector_chunk(
        patient_id=patient_id,
        source_type=SOURCE_KEY_EVENT,
        source_id=eid,
        content_text=content,
        embedding=vec,
      )
    except ValueError as e:
      return str(e)
  return None


def reindex_all_key_events(db: PatientDatabase, patient_id: str, limit: int = 500) -> Tuple[int, Optional[str]]:
  """为该患者所有关键事件重建 FTS5 与向量索引。"""
  try:
    db.sync_key_events_fts_for_patient(patient_id=patient_id)
  except Exception:
    pass
  events = db.list_key_events(patient_id=patient_id, limit=limit, offset=0)
  ids = [str(e.get("event_id")) for e in events if e.get("event_id")]
  if not ids:
    return 0, None
  err = index_key_events(db, patient_id, ids)
  return len(ids), err


def vector_search(
  db: PatientDatabase,
  patient_id: str,
  query: str,
  top_k: int = 8,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
  """返回 (结果列表, 错误信息)。结果含 score、chunk_id、source_type、source_id、content_text。"""
  q = (query or "").strip()
  if not q:
    return [], "empty_query"
  chunks = db.list_memory_vector_chunks_for_patient(patient_id=patient_id)
  if not chunks:
    logger.info("vector_search no_chunks patient_id=%s", patient_id)
    return [], None
  q_vecs, err = embed_texts([q])
  if err:
    logger.warning("vector_search embed failed patient_id=%s: %s", patient_id, err)
    return [], err
  qv = q_vecs[0]
  scored: List[Tuple[float, Dict[str, Any]]] = []
  for ch in chunks:
    emb = ch.get("embedding") or []
    if not emb:
      continue
    s = cosine_similarity(qv, emb)
    scored.append(
      (
        s,
        {
          "chunk_id": ch.get("chunk_id"),
          "source_type": ch.get("source_type"),
          "source_id": ch.get("source_id"),
          "content_text": ch.get("content_text"),
          "score": round(float(s), 6),
          "retrieval": "vector",
        },
      )
    )
  scored.sort(key=lambda x: x[0], reverse=True)
  out = [x[1] for x in scored[: max(1, top_k)]]
  logger.info(
    "vector_search patient_id=%s top_k=%s chunks=%s results=%s",
    patient_id,
    top_k,
    len(scored),
    len(out),
  )
  return out, None


def _fts_match_query(user_query: str) -> str:
  """构造 FTS5 MATCH 子句（OR 连接词元，兼容中英文）。"""
  q = (user_query or "").strip()
  if not q:
    return '""'
  tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", q)
  if not tokens:
    safe = q.replace('"', '""')
    return f'"{safe}"'
  parts: List[str] = []
  for t in tokens[:24]:
    parts.append('"' + t.replace('"', '""') + '"')
  return " OR ".join(parts)


def _rows_from_event_ids(
  db: PatientDatabase,
  patient_id: str,
  event_ids: List[str],
  chunks_by_source: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
  """补全仅有 FTS、尚无向量块的事件正文。"""
  missing = [e for e in event_ids if e and e not in chunks_by_source]
  if not missing:
    return chunks_by_source
  rows = db.get_key_events_by_ids(patient_id=patient_id, event_ids=missing)
  out = dict(chunks_by_source)
  for r in rows:
    eid = str(r.get("event_id") or "")
    if not eid:
      continue
    body = _key_event_text(r)
    out[eid] = {
      "chunk_id": None,
      "source_type": SOURCE_KEY_EVENT,
      "source_id": eid,
      "content_text": body,
      "embedding": [],
    }
  return out


def hybrid_search(
  db: PatientDatabase,
  patient_id: str,
  query: str,
  top_k: int = 8,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
  """FTS5（bm25 排序）与稠密向量检索做 RRF 融合；结果含 score（RRF）、retrieval=hybrid。"""
  q = (query or "").strip()
  if not q:
    return [], "empty_query"
  try:
    db.ensure_key_events_fts_aligned(patient_id)
  except Exception:
    pass

  cand = max(top_k * _CANDIDATE_MULT, top_k + 4, 16)
  dense, err = vector_search(db, patient_id, q, top_k=cand)
  if err and "QWEN_API_KEY" in err:
    dense, err = [], err
  elif err:
    dense = []

  fts_q = _fts_match_query(q)
  sparse: List[Dict[str, Any]] = []
  try:
    sparse = db.search_key_events_fts(patient_id=patient_id, fts_match_query=fts_q, limit=cand)
  except Exception:
    sparse = []

  if not dense and not sparse:
    if err:
      logger.warning("hybrid_search no results patient_id=%s err=%s", patient_id, err)
      return [], err
    logger.info("hybrid_search empty patient_id=%s", patient_id)
    return [], None

  logger.info(
    "hybrid_search patient_id=%s top_k=%s dense=%s sparse=%s",
    patient_id,
    top_k,
    len(dense),
    len(sparse),
  )

  rrf: Dict[str, float] = {}
  rk = float(_RRF_K)
  for rank, item in enumerate(dense, start=1):
    sid = str(item.get("source_id") or "")
    if sid:
      rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (rk + rank)
  for rank, item in enumerate(sparse, start=1):
    eid = str(item.get("event_id") or "")
    if eid:
      rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (rk + rank)

  merged_ids = [eid for eid, _ in sorted(rrf.items(), key=lambda x: -x[1])[: max(1, top_k)]]
  chunks = db.list_memory_vector_chunks_for_patient(patient_id=patient_id)
  by_source = {str(c.get("source_id") or ""): c for c in chunks if c.get("source_id")}
  by_source = _rows_from_event_ids(db, patient_id, merged_ids, by_source)

  out: List[Dict[str, Any]] = []
  for eid in merged_ids:
    ch = by_source.get(eid)
    if not ch:
      continue
    fusion = round(float(rrf.get(eid, 0.0)), 6)
    base = ch.get("content_text") or ""
    out.append(
      {
        "chunk_id": ch.get("chunk_id"),
        "source_type": ch.get("source_type") or SOURCE_KEY_EVENT,
        "source_id": eid,
        "content_text": base,
        "score": fusion,
        "fusion_score": fusion,
        "retrieval": "hybrid",
      }
    )
  logger.info("hybrid_search merged patient_id=%s out=%s", patient_id, len(out))
  return out, None


def fts_search_events(
  db: PatientDatabase,
  patient_id: str,
  query: str,
  top_k: int = 8,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
  """仅 FTS5，score 用 bm25 的负值归一（越大越好）。"""
  q = (query or "").strip()
  if not q:
    return [], "empty_query"
  try:
    db.ensure_key_events_fts_aligned(patient_id)
  except Exception:
    pass
  fts_q = _fts_match_query(q)
  try:
    sparse = db.search_key_events_fts(patient_id=patient_id, fts_match_query=fts_q, limit=max(top_k * 2, 16))
  except Exception:
    sparse = []
  if not sparse:
    logger.info("fts_search_events empty patient_id=%s", patient_id)
    return [], None
  slice_s = sparse[: max(1, top_k)]
  ids = [str(x["event_id"]) for x in slice_s]
  by_source = _rows_from_event_ids(db, patient_id, ids, {})
  bm25s = [float(x.get("bm25") or 0.0) for x in slice_s]
  if not bm25s:
    return [], None
  lo, hi = min(bm25s), max(bm25s)
  span = (hi - lo) or 1.0
  out: List[Dict[str, Any]] = []
  for row in slice_s:
    eid = str(row["event_id"])
    ch = by_source.get(eid)
    if not ch:
      continue
    b = float(row["bm25"] or 0.0)
    norm = 1.0 - (b - lo) / span
    out.append(
      {
        "chunk_id": ch.get("chunk_id"),
        "source_type": ch.get("source_type") or SOURCE_KEY_EVENT,
        "source_id": eid,
        "content_text": ch.get("content_text") or "",
        "score": round(norm, 6),
        "bm25": b,
        "retrieval": "fts",
      }
    )
  logger.info("fts_search_events patient_id=%s out=%s", patient_id, len(out))
  return out, None
