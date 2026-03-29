from backend.patient_db import PatientDatabase


def main() -> None:
  db = PatientDatabase()
  with db._conn() as conn:  # noqa: SLF001 - internal maintenance script
    conn.execute(
      """
      UPDATE medical_cases
      SET case_title = REPLACE(case_title, 'Follow-up case', '复诊随访病例'),
          diagnosis = CASE
            WHEN diagnosis = 'Chronic condition follow-up' THEN '慢性病随访'
            ELSE diagnosis
          END,
          description = REPLACE(description, 'Auto generated case for', '自动生成随访病例，患者编码')
      WHERE source = 'EMR'
      """
    )

    conn.execute(
      """
      UPDATE visit_records
      SET department = CASE
            WHEN department = 'General Medicine' THEN '全科门诊'
            ELSE department
          END,
          chief_complaint = CASE
            WHEN chief_complaint = 'Follow-up and symptom review' THEN '复诊评估与症状复盘'
            ELSE chief_complaint
          END,
          diagnosis = CASE
            WHEN diagnosis = 'Stable under treatment' THEN '病情稳定（治疗中）'
            ELSE diagnosis
          END,
          doctor = CASE
            WHEN doctor = 'Dr.Auto' THEN '自动化测试医生'
            ELSE doctor
          END,
          notes = REPLACE(notes, 'Auto generated visit for', '自动生成复诊记录，患者编码')
      WHERE source = 'EMR'
      """
    )

    conn.execute(
      """
      UPDATE case_qa
      SET query = CASE
            WHEN query = 'What should I monitor before next follow-up?' THEN '下次复诊前我需要重点监测什么？'
            ELSE query
          END,
          answer = CASE
            WHEN answer = 'Track key symptoms daily and keep medication adherence records before your next visit.'
              THEN '建议每日记录关键症状变化，并持续记录用药依从性，复诊时带上监测记录。'
            ELSE answer
          END,
          source = CASE
            WHEN source = 'auto_guideline' THEN '自动化测试知识库'
            ELSE source
          END
      """
    )

  print("convert_done")


if __name__ == "__main__":
  main()
