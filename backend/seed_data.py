from backend.patient_db import PatientDatabase


def seed() -> None:
  db = PatientDatabase()

  patients = [
    {
      "patient_code": "P1001",
      "name": "张三",
      "gender": "M",
      "birth_date": "1980-01-01",
      "phone": "13800000001",
      "id_number": "110101198001010011",
      "address": "北京市朝阳区",
      "raw_json": {"allergy": "青霉素", "chronic_disease": ["高血压"]},
    },
    {
      "patient_code": "P1002",
      "name": "李四",
      "gender": "F",
      "birth_date": "1975-05-20",
      "phone": "13800000002",
      "id_number": "110101197505200022",
      "address": "上海市浦东新区",
      "raw_json": {"allergy": "无", "chronic_disease": ["2型糖尿病"]},
    },
    {
      "patient_code": "P1003",
      "name": "王五",
      "gender": "M",
      "birth_date": "1992-11-12",
      "phone": "13800000003",
      "id_number": "110101199211120033",
      "address": "广州市天河区",
      "raw_json": {"allergy": "海鲜", "chronic_disease": []},
    },
    {
      "patient_code": "P1004",
      "name": "赵敏",
      "gender": "F",
      "birth_date": "1988-07-15",
      "phone": "13800000004",
      "id_number": "110101198807150044",
      "address": "杭州市西湖区",
      "raw_json": {"allergy": "无", "chronic_disease": ["甲状腺结节"]},
    },
    {
      "patient_code": "P1005",
      "name": "钱进",
      "gender": "M",
      "birth_date": "1969-03-02",
      "phone": "13800000005",
      "id_number": "110101196903020055",
      "address": "南京市鼓楼区",
      "raw_json": {"allergy": "阿司匹林", "chronic_disease": ["冠心病"]},
    },
    {
      "patient_code": "P1006",
      "name": "孙岚",
      "gender": "F",
      "birth_date": "1995-12-08",
      "phone": "13800000006",
      "id_number": "110101199512080066",
      "address": "成都市武侯区",
      "raw_json": {"allergy": "无", "chronic_disease": []},
    },
    {
      "patient_code": "P1007",
      "name": "周航",
      "gender": "M",
      "birth_date": "2001-09-18",
      "phone": "13800000007",
      "id_number": "110101200109180077",
      "address": "武汉市洪山区",
      "raw_json": {"allergy": "花粉", "chronic_disease": ["过敏性鼻炎"]},
    },
    {
      "patient_code": "P1008",
      "name": "吴楠",
      "gender": "F",
      "birth_date": "1978-04-26",
      "phone": "13800000008",
      "id_number": "110101197804260088",
      "address": "西安市雁塔区",
      "raw_json": {"allergy": "无", "chronic_disease": ["骨质疏松"]},
    },
  ]

  cases_by_code = {
    "P1001": [
      {
        "case_title": "高血压随访",
        "diagnosis": "原发性高血压",
        "description": "近半年血压波动，需调整生活方式与药物依从性。",
        "onset_date": "2025-08-15",
        "raw_json": {"severity": "medium", "bp_avg": "145/95"},
      },
      {
        "case_title": "上呼吸道感染",
        "diagnosis": "急性咽炎",
        "description": "咽痛伴低热 3 天。",
        "onset_date": "2026-02-20",
        "raw_json": {"severity": "low"},
      },
    ],
    "P1002": [
      {
        "case_title": "糖尿病复诊",
        "diagnosis": "2型糖尿病",
        "description": "近 3 月空腹血糖偏高，建议饮食管理与复查 HbA1c。",
        "onset_date": "2024-04-01",
        "raw_json": {"severity": "medium", "hba1c": "8.1%"},
      }
    ],
    "P1003": [
      {
        "case_title": "胃痛待查",
        "diagnosis": "慢性胃炎（待复核）",
        "description": "餐后上腹痛反复发作两周。",
        "onset_date": "2026-03-01",
        "raw_json": {"severity": "low", "recommend_exam": ["胃镜"]},
      }
    ],
    "P1004": [
      {
        "case_title": "甲状腺结节复查",
        "diagnosis": "甲状腺结节（TI-RADS 3）",
        "description": "建议半年超声随访，关注结节大小变化。",
        "onset_date": "2025-10-11",
        "raw_json": {"severity": "low"},
      }
    ],
    "P1005": [
      {
        "case_title": "冠心病随访",
        "diagnosis": "冠状动脉粥样硬化性心脏病",
        "description": "活动后胸闷偶发，需评估药物依从性和危险因素。",
        "onset_date": "2023-06-09",
        "raw_json": {"severity": "high", "risk_level": "medium-high"},
      }
    ],
    "P1006": [
      {
        "case_title": "偏头痛",
        "diagnosis": "偏头痛",
        "description": "周期性头痛伴畏光，睡眠不足时加重。",
        "onset_date": "2026-01-20",
        "raw_json": {"severity": "medium"},
      }
    ],
    "P1007": [
      {
        "case_title": "过敏性鼻炎",
        "diagnosis": "变应性鼻炎",
        "description": "季节性鼻塞流涕，晨起明显。",
        "onset_date": "2024-03-15",
        "raw_json": {"severity": "low"},
      }
    ],
    "P1008": [
      {
        "case_title": "骨密度下降",
        "diagnosis": "骨质疏松",
        "description": "骨密度检测提示T值降低，建议补钙与运动管理。",
        "onset_date": "2025-09-12",
        "raw_json": {"severity": "medium"},
      }
    ],
  }

  visits_by_code = {
    "P1001": [
      {
        "visit_date": "2026-03-10",
        "department": "心内科",
        "chief_complaint": "头晕、血压偏高",
        "diagnosis": "原发性高血压",
        "doctor": "赵医生",
        "notes": "建议低盐饮食，规律监测血压。",
        "source": "EMR",
        "raw_json": {"vitals": {"bp": "150/98", "hr": 82}},
      },
      {
        "visit_date": "2026-03-22",
        "department": "呼吸内科",
        "chief_complaint": "咽痛、轻咳",
        "diagnosis": "急性咽炎",
        "doctor": "陈医生",
        "notes": "多饮水，必要时复诊。",
        "source": "EMR",
        "raw_json": {"vitals": {"temp": 37.6}},
      },
    ],
    "P1002": [
      {
        "visit_date": "2026-03-18",
        "department": "内分泌科",
        "chief_complaint": "血糖控制不佳",
        "diagnosis": "2型糖尿病",
        "doctor": "刘医生",
        "notes": "调整二甲双胍剂量，建议一月后复查。",
        "source": "EMR",
        "raw_json": {"labs": {"fbg": 8.6, "hba1c": 8.1}},
      }
    ],
    "P1003": [
      {
        "visit_date": "2026-03-24",
        "department": "消化内科",
        "chief_complaint": "餐后腹痛",
        "diagnosis": "慢性胃炎（待复核）",
        "doctor": "孙医生",
        "notes": "建议胃镜检查，避免辛辣刺激。",
        "source": "EMR",
        "raw_json": {"vitals": {"temp": 36.7}, "plan": ["胃镜", "幽门螺杆菌检测"]},
      }
    ],
    "P1004": [
      {
        "visit_date": "2026-03-05",
        "department": "内分泌科",
        "chief_complaint": "复查甲状腺结节",
        "diagnosis": "甲状腺结节（TI-RADS 3）",
        "doctor": "郭医生",
        "notes": "建议半年复查超声，当前无需手术。",
        "source": "EMR",
        "raw_json": {"plan": ["6个月后超声复查"]},
      }
    ],
    "P1005": [
      {
        "visit_date": "2026-02-28",
        "department": "心内科",
        "chief_complaint": "活动后胸闷",
        "diagnosis": "冠心病",
        "doctor": "许医生",
        "notes": "建议规律服药，控制血脂，复查心电图。",
        "source": "EMR",
        "raw_json": {"vitals": {"bp": "138/86", "hr": 78}},
      }
    ],
    "P1006": [
      {
        "visit_date": "2026-03-12",
        "department": "神经内科",
        "chief_complaint": "反复头痛",
        "diagnosis": "偏头痛",
        "doctor": "梁医生",
        "notes": "建议规律作息，必要时按医嘱止痛。",
        "source": "EMR",
        "raw_json": {"trigger": ["睡眠不足", "压力"]},
      }
    ],
    "P1007": [
      {
        "visit_date": "2026-03-08",
        "department": "耳鼻喉科",
        "chief_complaint": "鼻塞流涕",
        "diagnosis": "过敏性鼻炎",
        "doctor": "胡医生",
        "notes": "建议减少过敏原暴露，按医嘱用药。",
        "source": "EMR",
        "raw_json": {"allergen": ["花粉"]},
      }
    ],
    "P1008": [
      {
        "visit_date": "2026-03-16",
        "department": "骨科",
        "chief_complaint": "腰背酸痛",
        "diagnosis": "骨质疏松",
        "doctor": "马医生",
        "notes": "建议补钙、维生素D及抗阻运动。",
        "source": "EMR",
        "raw_json": {"plan": ["补钙", "维生素D", "负重训练"]},
      }
    ],
  }

  qa_by_code = {
    "P1001": [
      {
        "query": "我的血压最近总是 150/95 左右，需要马上去急诊吗？",
        "answer": "若无胸痛、呼吸困难、神经系统症状，可先门诊复诊并连续监测血压；若收缩压持续>=180或出现明显不适，建议立即急诊。",
        "source": "doctor_knowledge_base",
        "tags": ["高血压", "急诊判断"],
      },
      {
        "query": "咽炎期间能不能继续运动？",
        "answer": "急性咽炎伴发热或明显咽痛时建议减少运动，待症状缓解后逐步恢复中低强度活动，注意补水与休息。",
        "source": "care_guideline",
        "tags": ["咽炎", "生活方式"],
      },
    ],
    "P1002": [
      {
        "query": "空腹血糖 8.6 算严重吗？",
        "answer": "空腹血糖 8.6 mmol/L 提示控制不达标，建议尽快复诊评估用药和饮食运动方案，并结合 HbA1c 判断近 3 个月控制情况。",
        "source": "endocrine_guideline",
        "tags": ["糖尿病", "血糖管理"],
      }
    ],
    "P1003": [
      {
        "query": "胃痛两周，必须做胃镜吗？",
        "answer": "若症状持续两周且反复发作，建议按医嘱完成胃镜与幽门螺杆菌检测，以明确病因并排除器质性病变。",
        "source": "digestive_guideline",
        "tags": ["消化内科", "检查建议"],
      }
    ],
    "P1004": [
      {
        "query": "甲状腺结节 TI-RADS 3 严重吗？",
        "answer": "通常属于低风险分级，建议按医嘱定期超声随访，关注结节变化即可。",
        "source": "endocrine_guideline",
        "tags": ["甲状腺", "随访"],
      }
    ],
    "P1005": [
      {
        "query": "冠心病日常最重要的管理是什么？",
        "answer": "规律服药、控制血压血脂、戒烟限酒和规律复查是核心，出现胸痛持续加重需及时就医。",
        "source": "cardio_guideline",
        "tags": ["冠心病", "慢病管理"],
      }
    ],
    "P1006": [
      {
        "query": "偏头痛发作时怎么缓解？",
        "answer": "建议先在安静避光环境休息，补充水分，按医嘱使用止痛药；若头痛突然剧烈或伴神经症状应急诊。",
        "source": "neuro_guideline",
        "tags": ["偏头痛", "急诊判断"],
      }
    ],
    "P1007": [
      {
        "query": "过敏性鼻炎会传染吗？",
        "answer": "不会传染，属于过敏反应。重点是识别并减少接触过敏原，同时规范用药。",
        "source": "ent_guideline",
        "tags": ["鼻炎", "过敏"],
      }
    ],
    "P1008": [
      {
        "query": "骨质疏松应该做哪些运动？",
        "answer": "建议在医生指导下进行负重和抗阻训练，并注意防跌倒；同时补充钙和维生素D。",
        "source": "ortho_guideline",
        "tags": ["骨质疏松", "运动管理"],
      }
    ],
  }

  inserted_patient_ids = {}
  for p in patients:
    pid = db.upsert_patient(p)
    inserted_patient_ids[p["patient_code"]] = pid

  case_count = 0
  visit_count = 0
  qa_count = 0

  for code, items in cases_by_code.items():
    pid = inserted_patient_ids[code]
    for item in items:
      db.add_medical_case(pid, item)
      case_count += 1

  for code, items in visits_by_code.items():
    pid = inserted_patient_ids[code]
    for item in items:
      db.add_visit_record(pid, item)
      visit_count += 1

  for code, items in qa_by_code.items():
    pid = inserted_patient_ids[code]
    for item in items:
      db.add_case_qa(
        patient_id=pid,
        query=item["query"],
        answer=item["answer"],
        source=item.get("source"),
        tags=item.get("tags"),
      )
      qa_count += 1

  print("seed_done")
  print("patients:", len(inserted_patient_ids))
  print("cases:", case_count)
  print("visits:", visit_count)
  print("qa:", qa_count)
  print("patient_ids:", inserted_patient_ids)


if __name__ == "__main__":
  seed()

