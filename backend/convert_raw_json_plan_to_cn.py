import json

from backend.patient_db import PatientDatabase


PLAN_MAP = {
  "continue medication": "继续用药",
  "recheck in 4 weeks": "4周后复查",
}


def _translate_plan(raw_json_text: str | None) -> str | None:
  if not raw_json_text:
    return raw_json_text
  try:
    obj = json.loads(raw_json_text)
  except Exception:
    return raw_json_text

  plan = obj.get("plan")
  if isinstance(plan, list):
    obj["plan"] = [PLAN_MAP.get(str(item), item) for item in plan]

  return json.dumps(obj, ensure_ascii=False)


def main() -> None:
  db = PatientDatabase()
  with db._conn() as conn:  # noqa: SLF001
    rows = conn.execute("SELECT visit_id, raw_json FROM visit_records").fetchall()
    changed = 0
    for row in rows:
      new_text = _translate_plan(row["raw_json"])
      if new_text != row["raw_json"]:
        conn.execute(
          "UPDATE visit_records SET raw_json = ? WHERE visit_id = ?",
          (new_text, row["visit_id"]),
        )
        changed += 1
  print(f"raw_json_plan_convert_done changed={changed}")


if __name__ == "__main__":
  main()
