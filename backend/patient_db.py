import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _phone_digits(raw: Optional[str]) -> str:
  if not raw:
    return ""
  return re.sub(r"\D", "", str(raw).strip())


def _now_iso() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sort_session_messages_newest_first(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """按 ts 降序：最近一条在最前（ISO8601/Z 可字典序比较）。"""
  if not messages:
    return messages

  def ts_key(m: Dict[str, Any]) -> str:
    return str(m.get("ts") or "")

  return sorted(messages, key=ts_key, reverse=True)


# 抽取/历史数据可能使用英文键；落库时统一为中文键名，便于展示与下游一致。
USER_PROFILE_KEY_EN_TO_ZH: Dict[str, str] = {
  "chronic_focus": "长期关注重点",
  "care_rhythm": "随访节奏",
  "notes": "备注说明",
  "health_focus": "健康关注点",
  "self_reported_symptoms": "自述症状",
  "follow_up_concerns": "随访关注点",
  "communication_style": "沟通风格",
}


def normalize_user_profile_keys(profile: Dict[str, Any]) -> Dict[str, Any]:
  if not profile:
    return {}
  out: Dict[str, Any] = {}
  for k, v in profile.items():
    nk = USER_PROFILE_KEY_EN_TO_ZH.get(k, k)
    out[nk] = v
  return out


class PatientDatabase:
  def __init__(self, db_path: Optional[str] = None) -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    self.db_path = db_path or str(data_dir / "patient_agent.db")
    self._init_db()

  def _conn(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    return conn

  def _init_db(self) -> None:
    with self._conn() as conn:
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
          patient_id TEXT PRIMARY KEY,
          patient_code TEXT UNIQUE,
          name TEXT,
          gender TEXT,
          birth_date TEXT,
          phone TEXT,
          id_number TEXT,
          address TEXT,
          raw_json TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
      )
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS medical_cases (
          case_id TEXT PRIMARY KEY,
          patient_id TEXT NOT NULL,
          case_title TEXT,
          diagnosis TEXT,
          description TEXT,
          onset_date TEXT,
          source TEXT,
          raw_json TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          record_key TEXT UNIQUE,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS visit_records (
          visit_id TEXT PRIMARY KEY,
          patient_id TEXT NOT NULL,
          visit_date TEXT,
          department TEXT,
          chief_complaint TEXT,
          diagnosis TEXT,
          doctor TEXT,
          notes TEXT,
          source TEXT,
          raw_json TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          record_key TEXT UNIQUE,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS case_qa (
          qa_id TEXT PRIMARY KEY,
          patient_id TEXT NOT NULL,
          case_id TEXT,
          query TEXT NOT NULL,
          answer TEXT NOT NULL,
          source TEXT,
          tags TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_phone ON patients(phone)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_patient ON medical_cases(patient_id)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_vr_patient ON visit_records(patient_id)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_patient ON case_qa(patient_id)")
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_settings (
          patient_id TEXT PRIMARY KEY,
          preferences TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_memory (
          session_id TEXT PRIMARY KEY,
          patient_id TEXT NOT NULL,
          messages TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_key_events (
          event_id TEXT PRIMARY KEY,
          patient_id TEXT NOT NULL,
          title TEXT,
          summary TEXT NOT NULL,
          source TEXT,
          event_date TEXT,
          confidence REAL,
          raw_json TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_user_profile (
          patient_id TEXT PRIMARY KEY,
          profile_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute("CREATE INDEX IF NOT EXISTS idx_session_memory_patient ON session_memory(patient_id)")
      conn.execute("CREATE INDEX IF NOT EXISTS idx_mke_patient ON memory_key_events(patient_id)")
      conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vector_chunks (
          chunk_id TEXT PRIMARY KEY,
          patient_id TEXT NOT NULL,
          source_type TEXT NOT NULL,
          source_id TEXT NOT NULL,
          content_text TEXT NOT NULL,
          embedding_json TEXT NOT NULL,
          dim INTEGER,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
        """
      )
      conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_mvc_patient_source
        ON memory_vector_chunks(patient_id, source_type, source_id)
        """
      )
      conn.execute("CREATE INDEX IF NOT EXISTS idx_mvc_patient ON memory_vector_chunks(patient_id)")
      conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_key_events_fts USING fts5(
          event_id UNINDEXED,
          patient_id UNINDEXED,
          body,
          tokenize = 'unicode61'
        )
        """
      )

  def _to_dict(self, row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
      return None
    obj = dict(row)
    for key in ("raw_json", "tags"):
      if key in obj and obj[key]:
        try:
          obj[key] = json.loads(obj[key])
        except Exception:
          pass
    return obj

  def _to_list(self, rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [self._to_dict(r) for r in rows if r is not None]  # type: ignore[arg-type]

  def list_patients(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with self._conn() as conn:
      rows = conn.execute(
        "SELECT * FROM patients ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
      ).fetchall()
    return self._to_list(rows)

  def get_patient(self, patient_id: Optional[str] = None, patient_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with self._conn() as conn:
      if patient_id:
        row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
      elif patient_code:
        row = conn.execute("SELECT * FROM patients WHERE patient_code = ?", (patient_code,)).fetchone()
      else:
        raise ValueError("patient_id or patient_code is required")
    return self._to_dict(row)

  def get_patient_by_phone(self, phone_digits: str) -> Optional[Dict[str, Any]]:
    """按院内登记手机号匹配（入参为纯数字 11 位；与库中号码去空格/横线后比较）。"""
    if not phone_digits or len(phone_digits) != 11 or not phone_digits.startswith("1"):
      return None
    with self._conn() as conn:
      rows = conn.execute(
        "SELECT * FROM patients WHERE phone IS NOT NULL AND trim(phone) != ''"
      ).fetchall()
    for r in rows:
      d = self._to_dict(r)
      if _phone_digits(d.get("phone")) == phone_digits:
        return d
    return None

  def upsert_patient(self, patient: Dict[str, Any]) -> str:
    patient_code = patient.get("patient_code")
    patient_id = patient.get("patient_id")
    if not patient_id and patient_code:
      existed = self.get_patient(patient_code=patient_code)
      if existed:
        patient_id = existed.get("patient_id")
    patient_id = patient_id or str(uuid.uuid4())
    now = _now_iso()
    raw_json = patient.get("raw_json")
    with self._conn() as conn:
      conn.execute(
        """
        INSERT INTO patients (
          patient_id, patient_code, name, gender, birth_date, phone, id_number, address, raw_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
          patient_code=excluded.patient_code,
          name=excluded.name,
          gender=excluded.gender,
          birth_date=excluded.birth_date,
          phone=excluded.phone,
          id_number=excluded.id_number,
          address=excluded.address,
          raw_json=excluded.raw_json,
          updated_at=excluded.updated_at
        """,
        (
          patient_id,
          patient.get("patient_code"),
          patient.get("name"),
          patient.get("gender"),
          patient.get("birth_date"),
          patient.get("phone"),
          patient.get("id_number"),
          patient.get("address"),
          json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
          now,
          now,
        ),
      )
      if patient.get("patient_code"):
        conn.execute(
          """
          UPDATE patients
          SET patient_id = ?, name = COALESCE(?, name), gender = COALESCE(?, gender),
              birth_date = COALESCE(?, birth_date), phone = COALESCE(?, phone),
              id_number = COALESCE(?, id_number), address = COALESCE(?, address),
              raw_json = COALESCE(?, raw_json), updated_at = ?
          WHERE patient_code = ? AND patient_id != ?
          """,
          (
            patient_id,
            patient.get("name"),
            patient.get("gender"),
            patient.get("birth_date"),
            patient.get("phone"),
            patient.get("id_number"),
            patient.get("address"),
            json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
            now,
            patient.get("patient_code"),
            patient_id,
          ),
        )
    return patient_id

  def _build_case_record_key(self, patient_id: str, case: Dict[str, Any]) -> str:
    return "|".join(
      [
        patient_id,
        str(case.get("onset_date") or ""),
        str(case.get("case_title") or ""),
        str(case.get("diagnosis") or ""),
      ]
    )

  def add_medical_case(self, patient_id: str, case: Dict[str, Any]) -> str:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    now = _now_iso()
    case_id = case.get("case_id") or str(uuid.uuid4())
    raw_json = case.get("raw_json")
    record_key = case.get("record_key") or self._build_case_record_key(patient_id, case)
    with self._conn() as conn:
      conn.execute(
        """
        INSERT INTO medical_cases (
          case_id, patient_id, case_title, diagnosis, description, onset_date, source, raw_json, created_at, updated_at, record_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key) DO UPDATE SET
          case_title=excluded.case_title,
          diagnosis=excluded.diagnosis,
          description=excluded.description,
          onset_date=excluded.onset_date,
          source=excluded.source,
          raw_json=excluded.raw_json,
          updated_at=excluded.updated_at
        """,
        (
          case_id,
          patient_id,
          case.get("case_title"),
          case.get("diagnosis"),
          case.get("description"),
          case.get("onset_date"),
          case.get("source"),
          json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
          now,
          now,
          record_key,
        ),
      )
      row = conn.execute("SELECT case_id FROM medical_cases WHERE record_key = ?", (record_key,)).fetchone()
    return str(row["case_id"]) if row else case_id

  def list_medical_cases(self, patient_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    with self._conn() as conn:
      rows = conn.execute(
        """
        SELECT * FROM medical_cases
        WHERE patient_id = ?
        ORDER BY COALESCE(onset_date, created_at) DESC, updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (patient_id, limit, offset),
      ).fetchall()
    return self._to_list(rows)

  def _build_visit_record_key(self, patient_id: str, visit: Dict[str, Any]) -> str:
    return "|".join(
      [
        patient_id,
        str(visit.get("visit_date") or ""),
        str(visit.get("department") or ""),
        str(visit.get("diagnosis") or ""),
        str(visit.get("doctor") or ""),
      ]
    )

  def add_visit_record(self, patient_id: str, visit: Dict[str, Any]) -> str:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    now = _now_iso()
    visit_id = visit.get("visit_id") or str(uuid.uuid4())
    raw_json = visit.get("raw_json")
    record_key = visit.get("record_key") or self._build_visit_record_key(patient_id, visit)
    with self._conn() as conn:
      conn.execute(
        """
        INSERT INTO visit_records (
          visit_id, patient_id, visit_date, department, chief_complaint, diagnosis, doctor, notes, source, raw_json, created_at, updated_at, record_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key) DO UPDATE SET
          visit_date=excluded.visit_date,
          department=excluded.department,
          chief_complaint=excluded.chief_complaint,
          diagnosis=excluded.diagnosis,
          doctor=excluded.doctor,
          notes=excluded.notes,
          source=excluded.source,
          raw_json=excluded.raw_json,
          updated_at=excluded.updated_at
        """,
        (
          visit_id,
          patient_id,
          visit.get("visit_date"),
          visit.get("department"),
          visit.get("chief_complaint"),
          visit.get("diagnosis"),
          visit.get("doctor"),
          visit.get("notes"),
          visit.get("source"),
          json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
          now,
          now,
          record_key,
        ),
      )
      row = conn.execute("SELECT visit_id FROM visit_records WHERE record_key = ?", (record_key,)).fetchone()
    return str(row["visit_id"]) if row else visit_id

  def list_visit_records(self, patient_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    with self._conn() as conn:
      rows = conn.execute(
        """
        SELECT * FROM visit_records
        WHERE patient_id = ?
        ORDER BY COALESCE(visit_date, created_at) DESC, updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (patient_id, limit, offset),
      ).fetchall()
    return self._to_list(rows)

  def add_case_qa(
    self,
    patient_id: str,
    query: str,
    answer: str,
    case_id: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
  ) -> str:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    now = _now_iso()
    qa_id = str(uuid.uuid4())
    with self._conn() as conn:
      conn.execute(
        """
        INSERT INTO case_qa (qa_id, patient_id, case_id, query, answer, source, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          qa_id,
          patient_id,
          case_id,
          query,
          answer,
          source,
          json.dumps(tags or [], ensure_ascii=False),
          now,
          now,
        ),
      )
    return qa_id

  def search_case_qa(
    self,
    patient_id: str,
    query: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
  ) -> List[Dict[str, Any]]:
    with self._conn() as conn:
      if query and query.strip():
        like = f"%{query.strip()}%"
        rows = conn.execute(
          """
          SELECT * FROM case_qa
          WHERE patient_id = ? AND (query LIKE ? OR answer LIKE ?)
          ORDER BY updated_at DESC
          LIMIT ? OFFSET ?
          """,
          (patient_id, like, like, limit, offset),
        ).fetchall()
      else:
        rows = conn.execute(
          """
          SELECT * FROM case_qa
          WHERE patient_id = ?
          ORDER BY updated_at DESC
          LIMIT ? OFFSET ?
          """,
          (patient_id, limit, offset),
        ).fetchall()
    return self._to_list(rows)

  def get_patient_full(self, patient_id: str) -> Optional[Dict[str, Any]]:
    patient = self.get_patient(patient_id=patient_id)
    if not patient:
      return None
    return {
      "patient": patient,
      "medical_cases": self.list_medical_cases(patient_id=patient_id, limit=200, offset=0),
      "visit_records": self.list_visit_records(patient_id=patient_id, limit=200, offset=0),
      "case_qa": self.search_case_qa(patient_id=patient_id, query=None, limit=200, offset=0),
    }

  def get_memory_settings(self, patient_id: str) -> Dict[str, Any]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      row = conn.execute(
        "SELECT * FROM memory_settings WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
    if not row:
      return {
        "patient_id": patient_id,
        "preferences": {},
        "created_at": None,
        "updated_at": None,
      }
    obj = dict(row)
    try:
      obj["preferences"] = json.loads(obj["preferences"]) if obj.get("preferences") else {}
    except Exception:
      obj["preferences"] = {}
    return obj

  def put_memory_settings(self, patient_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    now = _now_iso()
    prefs_json = json.dumps(preferences or {}, ensure_ascii=False)
    with self._conn() as conn:
      row = conn.execute(
        "SELECT patient_id FROM memory_settings WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
      if row:
        conn.execute(
          """
          UPDATE memory_settings
          SET preferences = ?, updated_at = ?
          WHERE patient_id = ?
          """,
          (prefs_json, now, patient_id),
        )
      else:
        conn.execute(
          """
          INSERT INTO memory_settings (patient_id, preferences, created_at, updated_at)
          VALUES (?, ?, ?, ?)
          """,
          (patient_id, prefs_json, now, now),
        )
    return self.get_memory_settings(patient_id=patient_id)

  def patch_memory_settings(self, patient_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
    current = self.get_memory_settings(patient_id=patient_id)
    merged = {**(current.get("preferences") or {}), **(preferences or {})}
    return self.put_memory_settings(patient_id=patient_id, preferences=merged)

  def append_session_message(
    self,
    patient_id: str,
    session_id: str,
    role: str,
    content: str,
    extras: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    now = _now_iso()
    entry: Dict[str, Any] = {"role": role, "content": content, "ts": now}
    if extras:
      entry["extras"] = extras
    with self._conn() as conn:
      row = conn.execute(
        "SELECT messages, created_at FROM session_memory WHERE session_id = ? AND patient_id = ?",
        (session_id, patient_id),
      ).fetchone()
      if row:
        try:
          msgs = json.loads(row["messages"] or "[]")
        except Exception:
          msgs = []
        msgs.append(entry)
        conn.execute(
          "UPDATE session_memory SET messages = ?, updated_at = ? WHERE session_id = ?",
          (json.dumps(msgs, ensure_ascii=False), now, session_id),
        )
        created_at = row["created_at"]
      else:
        msgs = [entry]
        conn.execute(
          """
          INSERT INTO session_memory (session_id, patient_id, messages, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?)
          """,
          (session_id, patient_id, json.dumps(msgs, ensure_ascii=False), now, now),
        )
        created_at = now
    return self.get_session_memory(
      patient_id=patient_id, session_id=session_id, newest_first=True
    )

  def get_session_memory(
    self,
    patient_id: str,
    session_id: str,
    *,
    newest_first: bool = False,
  ) -> Dict[str, Any]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      row = conn.execute(
        "SELECT * FROM session_memory WHERE session_id = ? AND patient_id = ?",
        (session_id, patient_id),
      ).fetchone()
    if not row:
      return {
        "session_id": session_id,
        "patient_id": patient_id,
        "messages": [],
        "created_at": None,
        "updated_at": None,
      }
    obj = dict(row)
    try:
      obj["messages"] = json.loads(obj["messages"] or "[]")
    except Exception:
      obj["messages"] = []
    if newest_first and obj["messages"]:
      obj["messages"] = _sort_session_messages_newest_first(obj["messages"])
    return obj

  def delete_session_memory(self, patient_id: str, session_id: str) -> None:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      conn.execute(
        "DELETE FROM session_memory WHERE session_id = ? AND patient_id = ?",
        (session_id, patient_id),
      )

  def insert_key_events(self, patient_id: str, events: List[Dict[str, Any]]) -> List[str]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    now = _now_iso()
    ids: List[str] = []
    with self._conn() as conn:
      for ev in events:
        eid = str(uuid.uuid4())
        raw = ev.get("raw") or ev
        title = ev.get("title")
        summary = ev.get("summary") or ev.get("title") or ""
        conn.execute(
          """
          INSERT INTO memory_key_events (
            event_id, patient_id, title, summary, source, event_date, confidence, raw_json, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            eid,
            patient_id,
            title,
            summary,
            ev.get("source") or "extracted",
            ev.get("event_date"),
            float(ev["confidence"]) if ev.get("confidence") is not None else None,
            json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else None,
            now,
            now,
          ),
        )
        try:
          self._insert_key_event_fts_row(conn, patient_id, eid, title, summary)
        except sqlite3.OperationalError:
          logging.getLogger(__name__).warning("fts_insert_skipped for event_id=%s", eid)
        ids.append(eid)
    return ids

  def list_key_events(
    self,
    patient_id: str,
    limit: int = 50,
    offset: int = 0,
  ) -> List[Dict[str, Any]]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      rows = conn.execute(
        """
        SELECT * FROM memory_key_events
        WHERE patient_id = ?
        ORDER BY COALESCE(event_date, created_at) DESC, updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (patient_id, limit, offset),
      ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
      d = dict(row)
      if d.get("raw_json"):
        try:
          d["raw_json"] = json.loads(d["raw_json"])
        except Exception:
          pass
      out.append(d)
    return out

  def get_key_events_by_ids(
    self,
    patient_id: str,
    event_ids: List[str],
  ) -> List[Dict[str, Any]]:
    """按插入顺序返回事件（与 event_ids 顺序一致）。"""
    if not event_ids:
      return []
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    placeholders = ",".join("?" * len(event_ids))
    with self._conn() as conn:
      rows = conn.execute(
        f"""
        SELECT * FROM memory_key_events
        WHERE patient_id = ? AND event_id IN ({placeholders})
        """,
        (patient_id, *event_ids),
      ).fetchall()
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
      d = dict(row)
      if d.get("raw_json"):
        try:
          d["raw_json"] = json.loads(d["raw_json"])
        except Exception:
          pass
      eid = d.get("event_id")
      if eid:
        by_id[str(eid)] = d
    out: List[Dict[str, Any]] = []
    for eid in event_ids:
      if eid in by_id:
        out.append(by_id[eid])
    return out

  def get_extracted_user_profile(self, patient_id: str) -> Dict[str, Any]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      row = conn.execute(
        "SELECT * FROM memory_user_profile WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
    if not row:
      return {
        "patient_id": patient_id,
        "profile": {},
        "created_at": None,
        "updated_at": None,
      }
    obj = dict(row)
    try:
      obj["profile"] = json.loads(obj.pop("profile_json") or "{}")
    except Exception:
      obj["profile"] = {}
    obj["profile"] = normalize_user_profile_keys(obj.get("profile") or {})
    return obj

  def merge_extracted_user_profile(self, patient_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    current = self.get_extracted_user_profile(patient_id=patient_id)
    cur_p = normalize_user_profile_keys(current.get("profile") or {})
    merged = {**cur_p, **normalize_user_profile_keys(profile or {})}
    now = _now_iso()
    pj = json.dumps(merged, ensure_ascii=False)
    with self._conn() as conn:
      row = conn.execute(
        "SELECT patient_id FROM memory_user_profile WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
      if row:
        conn.execute(
          "UPDATE memory_user_profile SET profile_json = ?, updated_at = ? WHERE patient_id = ?",
          (pj, now, patient_id),
        )
      else:
        conn.execute(
          """
          INSERT INTO memory_user_profile (patient_id, profile_json, created_at, updated_at)
          VALUES (?, ?, ?, ?)
          """,
          (patient_id, pj, now, now),
        )
    return self.get_extracted_user_profile(patient_id=patient_id)

  def upsert_memory_vector_chunk(
    self,
    patient_id: str,
    source_type: str,
    source_id: str,
    content_text: str,
    embedding: List[float],
  ) -> str:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    if not embedding:
      raise ValueError("empty_embedding")
    now = _now_iso()
    chunk_id = str(
      uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"memoryvec:{patient_id}:{source_type}:{source_id}",
      )
    )
    dim = len(embedding)
    ej = json.dumps(embedding, ensure_ascii=False)
    with self._conn() as conn:
      conn.execute(
        "DELETE FROM memory_vector_chunks WHERE patient_id = ? AND source_type = ? AND source_id = ?",
        (patient_id, source_type, source_id),
      )
      conn.execute(
        """
        INSERT INTO memory_vector_chunks (
          chunk_id, patient_id, source_type, source_id, content_text, embedding_json, dim, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, patient_id, source_type, source_id, content_text, ej, dim, now, now),
      )
    return chunk_id

  def list_memory_vector_chunks_for_patient(self, patient_id: str) -> List[Dict[str, Any]]:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      rows = conn.execute(
        """
        SELECT chunk_id, patient_id, source_type, source_id, content_text, embedding_json, dim, created_at, updated_at
        FROM memory_vector_chunks
        WHERE patient_id = ?
        ORDER BY updated_at DESC
        """,
        (patient_id,),
      ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
      d = dict(row)
      if d.get("embedding_json"):
        try:
          d["embedding"] = json.loads(d["embedding_json"])
        except Exception:
          d["embedding"] = []
      else:
        d["embedding"] = []
      del d["embedding_json"]
      out.append(d)
    return out

  def count_memory_vector_chunks_for_patient(self, patient_id: str) -> int:
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      row = conn.execute(
        "SELECT COUNT(*) AS c FROM memory_vector_chunks WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
    return int(row["c"]) if row else 0

  @staticmethod
  def _key_event_body(title: Optional[str], summary: str) -> str:
    t = (title or "").strip()
    s = (summary or "").strip()
    if t and s:
      return f"{t}\n{s}"
    return t or s

  def _insert_key_event_fts_row(
    self,
    conn: sqlite3.Connection,
    patient_id: str,
    event_id: str,
    title: Optional[str],
    summary: str,
  ) -> None:
    body = self._key_event_body(title, summary) or (summary or "")
    conn.execute(
      "INSERT INTO memory_key_events_fts(event_id, patient_id, body) VALUES (?, ?, ?)",
      (event_id, patient_id, body),
    )

  def sync_key_events_fts_for_patient(self, patient_id: str) -> None:
    """用 memory_key_events 全量重建该患者在 FTS5 中的行。"""
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      conn.execute("DELETE FROM memory_key_events_fts WHERE patient_id = ?", (patient_id,))
      rows = conn.execute(
        "SELECT event_id, patient_id, title, summary FROM memory_key_events WHERE patient_id = ?",
        (patient_id,),
      ).fetchall()
      for row in rows:
        eid = str(row["event_id"])
        self._insert_key_event_fts_row(
          conn,
          patient_id,
          eid,
          row["title"],
          row["summary"] or "",
        )

  def ensure_key_events_fts_aligned(self, patient_id: str) -> None:
    """若 FTS 行数少于关键事件，则自动同步（兼容升级前已有数据）。"""
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      row_m = conn.execute(
        "SELECT COUNT(*) AS c FROM memory_key_events WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
      row_f = conn.execute(
        "SELECT COUNT(*) AS c FROM memory_key_events_fts WHERE patient_id = ?",
        (patient_id,),
      ).fetchone()
    n_m = int(row_m["c"]) if row_m else 0
    n_f = int(row_f["c"]) if row_f else 0
    if n_m > n_f:
      logging.getLogger(__name__).info("key_events_fts backfill: patient_id=%s m=%s f=%s", patient_id, n_m, n_f)
      self.sync_key_events_fts_for_patient(patient_id)

  def search_key_events_fts(
    self,
    patient_id: str,
    fts_match_query: str,
    limit: int = 24,
  ) -> List[Dict[str, Any]]:
    """FTS5 + bm25；返回 event_id, bm25（越小越相关）。"""
    if not self.get_patient(patient_id=patient_id):
      raise ValueError("patient_not_found")
    with self._conn() as conn:
      try:
        rows = conn.execute(
          """
          SELECT event_id, bm25(memory_key_events_fts) AS bm25
          FROM memory_key_events_fts
          WHERE memory_key_events_fts MATCH ? AND patient_id = ?
          ORDER BY bm25 ASC
          LIMIT ?
          """,
          (fts_match_query, patient_id, limit),
        ).fetchall()
      except sqlite3.OperationalError as e:
        logging.getLogger(__name__).warning("fts_search_failed: %s", e)
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
      out.append({"event_id": str(row["event_id"]), "bm25": float(row["bm25"]) if row["bm25"] is not None else 0.0})
    return out
