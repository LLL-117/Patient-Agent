"""
短期会话 → 长期记忆：用量阈值触发与定时后台刷新。

环境变量（均为可选）：
- MEMORY_USAGE_EXTRACT_ENABLED：默认 true；0/false/off 关闭「消息数达阈值则抽取」。
- MEMORY_USAGE_EXTRACT_THRESHOLD：默认 20；自上次用量抽取以来，session 内**新增消息条数**达到该值则触发抽取。
- MEMORY_PERIODIC_EXTRACT_ENABLED：默认 false；1/true 开启定时扫描。
- MEMORY_PERIODIC_EXTRACT_INTERVAL_SECONDS：默认 86400（24h）；对同一会话，两次定时抽取之间的最小间隔。
- MEMORY_PERIODIC_EXTRACT_MIN_MESSAGES：默认 2；定时任务仅处理消息数不少于该值的会话。
- MEMORY_EXTRACT_DEBOUNCE_SECONDS：默认 120；同一 session 在窗口内不重复抽取（用量与定时共用）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .memory_extract import persist_extraction_result, run_extract_from_dialogue
from .patient_db import PatientDatabase, _now_iso

_log = logging.getLogger(__name__)

_extract_lock = asyncio.Lock()
_periodic_task: Optional[asyncio.Task] = None


def _env_bool(name: str, default: bool) -> bool:
  raw = (os.getenv(name) or "").strip().lower()
  if not raw:
    return default
  return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
  raw = (os.getenv(name) or "").strip()
  if not raw:
    return default
  try:
    return int(raw)
  except ValueError:
    return default


def _patient_basic(patient: Dict[str, Any]) -> Dict[str, Any]:
  return {
    "patient_id": patient.get("patient_id"),
    "patient_code": patient.get("patient_code"),
    "name": patient.get("name"),
    "gender": patient.get("gender"),
    "birth_date": patient.get("birth_date"),
  }


def _parse_iso_ts(s: Optional[str]) -> Optional[datetime]:
  if not s or not str(s).strip():
    return None
  t = str(s).strip().replace("Z", "+00:00")
  try:
    return datetime.fromisoformat(t)
  except Exception:
    return None


def run_dialogue_extract(
  db: PatientDatabase,
  patient_id: str,
  session_id: str,
) -> Optional[str]:
  """
  与 POST .../memory/extract-dialogue 相同的抽取与落库。
  成功返回 None；失败返回错误说明（仅记录日志）。
  """
  if not (os.getenv("QWEN_API_KEY", "").strip()):
    return "QWEN_API_KEY is not set"

  patient = db.get_patient(patient_id=patient_id)
  if not patient:
    return "patient_not_found"

  sess = db.get_session_memory(patient_id=patient_id, session_id=session_id, newest_first=False)
  dialogue_messages = sess.get("messages") or []
  if not dialogue_messages:
    return "no_messages"

  result = run_extract_from_dialogue(
    dialogue_messages=dialogue_messages,
    patient_basic=_patient_basic(patient),
  )
  if result.get("error"):
    return str(result.get("error"))

  try:
    persist_extraction_result(db, patient_id, result)
  except ValueError:
    return "patient_not_found"
  except Exception as e:
    return f"persist_failed: {e}"

  return None


async def _usage_extract_task(db: PatientDatabase, patient_id: str, session_id: str) -> None:
  async with _extract_lock:
    err = await asyncio.to_thread(run_dialogue_extract, db, patient_id, session_id)
  now = _now_iso()
  if err:
    _log.warning(
      "usage_extract failed patient_id=%s session_id=%s err=%s",
      patient_id,
      session_id,
      err,
    )
    return
  try:
    sess2 = db.get_session_memory(patient_id=patient_id, session_id=session_id, newest_first=False)
    n = len(sess2.get("messages") or [])
    db.upsert_session_extract_state(
      patient_id,
      session_id,
      last_usage_extract_msg_len=n,
      last_any_extract_at=now,
    )
  except Exception as e:
    _log.warning("usage_extract state update: %s", e)
    return

  _log.info(
    "usage_extract ok patient_id=%s session_id=%s messages=%s",
    patient_id,
    session_id,
    n,
  )


def schedule_usage_refresh_if_needed(db: PatientDatabase, patient_id: str, session_id: str) -> None:
  """
  在写入本轮 assistant 消息之后调用（同步或 async 内均可）。
  若当前会话自上次用量抽取以来新增消息数 ≥ 阈值，则调度后台抽取任务。
  """
  if not _env_bool("MEMORY_USAGE_EXTRACT_ENABLED", True):
    return
  thr = _env_int("MEMORY_USAGE_EXTRACT_THRESHOLD", 20)
  if thr <= 0:
    return

  try:
    sess = db.get_session_memory(patient_id=patient_id, session_id=session_id, newest_first=False)
    msg_count = len(sess.get("messages") or [])
  except Exception as e:
    _log.warning("usage_refresh read session: %s", e)
    return

  state = db.get_session_extract_state(patient_id, session_id)
  last_len = int(state.get("last_usage_extract_msg_len") or 0)
  if msg_count - last_len < thr:
    return

  debounce = max(0, _env_int("MEMORY_EXTRACT_DEBOUNCE_SECONDS", 120))
  last_any = _parse_iso_ts(state.get("last_any_extract_at"))
  if debounce and last_any is not None:
    try:
      la = last_any if last_any.tzinfo else last_any.replace(tzinfo=timezone.utc)
      delta = datetime.now(timezone.utc) - la
      if delta.total_seconds() < debounce:
        return
    except Exception:
      pass

  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    _log.warning("usage_refresh: no running event loop, skip scheduling")
    return

  loop.create_task(_usage_extract_task(db, patient_id, session_id))


async def _periodic_refresh_loop(db: PatientDatabase) -> None:
  interval = max(60, _env_int("MEMORY_PERIODIC_EXTRACT_INTERVAL_SECONDS", 86400))
  min_msg = max(1, _env_int("MEMORY_PERIODIC_EXTRACT_MIN_MESSAGES", 2))
  debounce = max(0, _env_int("MEMORY_EXTRACT_DEBOUNCE_SECONDS", 120))

  while True:
    try:
      if _env_bool("MEMORY_PERIODIC_EXTRACT_ENABLED", False) and (
        os.getenv("QWEN_API_KEY", "").strip()
      ):
        rows = db.list_session_memory_rows()
      else:
        rows = []
    except Exception as e:
      _log.warning("periodic_extract list sessions: %s", e)
      rows = []

    now = datetime.now(timezone.utc)
    for row in rows:
      pid = row["patient_id"]
      sid = row["session_id"]
      mc = int(row.get("message_count") or 0)
      if mc < min_msg:
        continue

      st = db.get_session_extract_state(pid, sid)
      last_p = _parse_iso_ts(st.get("last_periodic_extract_at"))
      last_any = _parse_iso_ts(st.get("last_any_extract_at"))

      if debounce and last_any is not None:
        try:
          la = last_any if last_any.tzinfo else last_any.replace(tzinfo=timezone.utc)
          if (now - la).total_seconds() < debounce:
            continue
        except Exception:
          pass

      due = False
      if last_p is None:
        due = True
      else:
        try:
          lp = last_p if last_p.tzinfo else last_p.replace(tzinfo=timezone.utc)
          if (now - lp).total_seconds() >= interval:
            due = True
        except Exception:
          due = True

      if not due:
        continue

      async with _extract_lock:
        err = await asyncio.to_thread(run_dialogue_extract, db, pid, sid)
      ts = _now_iso()
      if err:
        _log.warning("periodic_extract failed patient_id=%s session_id=%s err=%s", pid, sid, err)
        continue

      try:
        sess3 = db.get_session_memory(patient_id=pid, session_id=sid, newest_first=False)
        nlen = len(sess3.get("messages") or [])
        db.upsert_session_extract_state(
          pid,
          sid,
          last_usage_extract_msg_len=nlen,
          last_periodic_extract_at=ts,
          last_any_extract_at=ts,
        )
      except Exception as e:
        _log.warning("periodic_extract state: %s", e)
        continue

      _log.info("periodic_extract ok patient_id=%s session_id=%s messages=%s", pid, sid, mc)

    await asyncio.sleep(interval)


def start_periodic_refresh_background(db: PatientDatabase) -> None:
  global _periodic_task
  if not _env_bool("MEMORY_PERIODIC_EXTRACT_ENABLED", False):
    _log.info("periodic memory extract disabled (MEMORY_PERIODIC_EXTRACT_ENABLED)")
    return

  async def _runner() -> None:
    try:
      await _periodic_refresh_loop(db)
    except asyncio.CancelledError:
      raise
    except Exception as e:
      _log.exception("periodic_refresh_loop crashed: %s", e)

  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    return

  if _periodic_task is not None and not _periodic_task.done():
    return

  _periodic_task = loop.create_task(_runner())
  _log.info(
    "periodic memory extract started (interval=%ss)",
    max(60, _env_int("MEMORY_PERIODIC_EXTRACT_INTERVAL_SECONDS", 86400)),
  )


def stop_periodic_refresh_background() -> None:
  global _periodic_task
  if _periodic_task is None or _periodic_task.done():
    return
  _periodic_task.cancel()
  _log.info("periodic memory extract task cancelled")
