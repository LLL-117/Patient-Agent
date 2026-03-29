from backend.patient_db import PatientDatabase


def main() -> None:
  db = PatientDatabase()
  codes = [f"P100{i}" for i in range(1, 9)]
  summary: list[tuple[str, int, int, int]] = []

  for idx, code in enumerate(codes, start=1):
    patient = db.get_patient(patient_code=code)
    if not patient:
      pid = db.upsert_patient(
        {
          "patient_code": code,
          "name": f"TestUser{idx}",
          "gender": "M" if idx % 2 else "F",
          "birth_date": f"198{idx % 10}-0{(idx % 9) + 1}-15",
          "phone": f"139000000{idx:02d}",
          "address": "Test City",
          "raw_json": {"source": "auto_seed_batch"},
        }
      )
      patient = db.get_patient(patient_id=pid)
      if not patient:
        raise RuntimeError(f"failed_to_create_patient: {code}")

    patient_id = patient["patient_id"]

    db.add_medical_case(
      patient_id,
      {
        "case_title": f"Follow-up case {idx}",
        "diagnosis": "Chronic condition follow-up",
        "description": f"Auto generated case for {code}.",
        "onset_date": f"2026-05-{10 + idx:02d}",
        "source": "EMR",
        "raw_json": {"severity": "medium", "batch": "2026-03-27"},
      },
    )

    db.add_visit_record(
      patient_id,
      {
        "visit_date": f"2026-05-{18 + idx:02d}",
        "department": "General Medicine",
        "chief_complaint": "Follow-up and symptom review",
        "diagnosis": "Stable under treatment",
        "doctor": "Dr.Auto",
        "notes": f"Auto generated visit for {code}.",
        "source": "EMR",
        "raw_json": {"plan": ["continue medication", "recheck in 4 weeks"], "batch": "2026-03-27"},
      },
    )

    db.add_case_qa(
      patient_id=patient_id,
      query="What should I monitor before next follow-up?",
      answer="Track key symptoms daily and keep medication adherence records before your next visit.",
      source="auto_guideline",
      tags=["follow_up", "self_monitoring", "batch_2026_03_27"],
    )

    full = db.get_patient_full(patient_id)
    if not full:
      raise RuntimeError(f"failed_to_fetch_full: {code}")
    summary.append((code, len(full["medical_cases"]), len(full["visit_records"]), len(full["case_qa"])))

  print("batch_seed_done")
  for row in summary:
    print(f"{row[0]} cases={row[1]} visits={row[2]} qa={row[3]}")


if __name__ == "__main__":
  main()
