from typing import List, Dict, Any

class CONFIG_DATA:
    TARGET_COLS: List[str] = [
        "SBD", # Mã số thí sinh
        "MATH", # Toán
        "LITERATURE", # Ngữ văn
        "PHYSICS", # Vật lý
        "CHEMISTRY", # Hóa học
        "BIOLOGY", # Sinh học
        "HISTORY", # Lịch sử
        "GEOGRAPHY", # Địa lý
        "CIVIC_EDUCATION", # Giáo dục công dân
        "FOREIGN_LANGUAGE", # Điểm ngoại ngữ
        "FOREIGN_LANGUAGE_CODE" # Mã ngoại ngữ
    ]

    map_2020: Dict[str, str] = {
        "student_id": "SBD",
        "mathematics": "MATH",
        "literature": "LITERATURE",
        "physics": "PHYSICS",
        "chemistry": "CHEMISTRY",
        "biology": "BIOLOGY",
        "history": "HISTORY",
        "geography": "GEOGRAPHY",
        "civic_education": "CIVIC_EDUCATION",
        "foreign_language_score": "FOREIGN_LANGUAGE",
        "foreign_language_code": "FOREIGN_LANGUAGE_CODE"
    }

    map_2021: Dict[str, str] = {
        "id_examinee": "SBD",
        "math": "MATH",
        "literature": "LITERATURE",
        "physics": "PHYSICS",
        "chemistry": "CHEMISTRY",
        "biology": "BIOLOGY",
        "history": "HISTORY",
        "geography": "GEOGRAPHY",
        "civic_education": "CIVIC_EDUCATION"
    }

    map_2022_2023_2024: Dict[str, str] = {
        "sbd": "SBD",
        "toan": "MATH",
        "ngu_van": "LITERATURE",
        "vat_li": "PHYSICS",
        "hoa_hoc": "CHEMISTRY",
        "sinh_hoc": "BIOLOGY",
        "lich_su": "HISTORY",
        "dia_li": "GEOGRAPHY",
        "gdcd": "CIVIC_EDUCATION",
        "ngoai_ngu": "FOREIGN_LANGUAGE",
        "ma_ngoai_ngu": "FOREIGN_LANGUAGE_CODE"
    }

    # Cấu hình các tổ hợp xét tuyển
    SCORE_FORMULAS: Dict[str, Dict[str, Any]] = {
        "A00": {
            "cols": ["MATH", "PHYSICS", "CHEMISTRY"], 
            "lang_req": None
        },
        "B00": {
            "cols": ["MATH", "CHEMISTRY", "BIOLOGY"], 
            "lang_req": None
        },
        "A01": {
            "cols": ["MATH", "PHYSICS", "FOREIGN_LANGUAGE"], 
            "lang_req": "N1"
        }, # N1: Tiếng Anh
        "D01": {
            "cols": ["MATH", "LITERATURE", "FOREIGN_LANGUAGE"], 
            "lang_req": "N1"
        },
        "D07": {
            "cols": ["MATH", "CHEMISTRY", "FOREIGN_LANGUAGE"], 
            "lang_req": "N1"
        },
        "D29": {
            "cols": ["MATH", "PHYSICS", "FOREIGN_LANGUAGE"], 
            "lang_req": "N3"
        }, # N3: Tiếng Pháp
        "D24": {
            "cols": ["MATH", "CHEMISTRY", "FOREIGN_LANGUAGE"], 
            "lang_req": "N3"
        }
    }

    # Thống kê điểm thi THPTQG qua các năm
    EXAM_STATS_DETAILED: Dict[int, Dict[str, Dict[str, float]]] = {
        2020: {
            "A00": {"mean": 21.4471, "std": 3.3425},
            "A01": {"mean": 20.0417, "std": 3.3084},
            "B00": {"mean": 20.3388, "std": 3.0812},
            "D01": {"mean": 18.1417, "std": 3.7811},
            "D07": {"mean": 20.0117, "std": 3.1333},
            "D24": {"mean": 22.8503, "std": 2.8891},
            "D29": {"mean": 22.3681, "std": 2.8403},
            "OTHER": {"mean": 19.5364, "std": 3.6661}
        },
        2021: {
            "A00": {"mean": 21.0262, "std": 3.1863},
            "A01": {"mean": 21.1029, "std": 3.4455},
            "B00": {"mean": 19.9892, "std": 3.0890},
            "D01": {"mean": 19.2666, "std": 4.1166},
            "D07": {"mean": 21.1338, "std": 3.2841},
            "D24": {"mean": 21.9896, "std": 3.0011},
            "D29": {"mean": 21.4177, "std": 3.0370},
            "OTHER": {"mean": 20.2071, "std": 3.7104}
        },
        2022: {
            "A00": {"mean": 21.0955, "std": 3.2378},
            "A01": {"mean": 20.2909, "std": 3.3396},
            "B00": {"mean": 19.4039, "std": 3.1555},
            "D01": {"mean": 18.4381, "std": 3.8846},
            "D07": {"mean": 20.2397, "std": 3.2064},
            "OTHER": {"mean": 19.5196, "std": 3.6548}
        },
        2023: {
            "A00": {"mean": 20.7745, "std": 3.0941},
            "A01": {"mean": 20.2743, "std": 3.3399},
            "B00": {"mean": 20.6047, "std": 2.7763},
            "D01": {"mean": 18.8891, "std": 3.8137},
            "D07": {"mean": 20.4216, "std": 3.0619},
            "D24": {"mean": 20.8364, "std": 3.3689},
            "D29": {"mean": 20.2285, "std": 3.5744},
            "OTHER": {"mean": 19.8594, "std": 3.4893}
        },
        2024: {
            "A00": {"mean": 20.9046, "std": 3.3804},
            "A01": {"mean": 20.4724, "std": 3.3509},
            "B00": {"mean": 20.5311, "std": 2.9818},
            "D01": {"mean": 19.4939, "std": 3.6232},
            "D07": {"mean": 20.4510, "std": 3.1120},
            "D24": {"mean": 21.8162, "std": 3.2930},
            "D29": {"mean": 20.9934, "std": 3.5598},
            "OTHER": {"mean": 20.1512, "std": 3.4281}
        },
    }

    # Thứ tự học kỳ để sắp xếp dữ liệu chuỗi thời gian
    SEMESTER_ORDER = [
        "HK1 2020-2021", "HK2 2020-2021",
        "HK1 2021-2022", "HK2 2021-2022",
        "HK1 2022-2023", "HK2 2022-2023",
        "HK1 2023-2024", "HK2 2023-2024",
        "HK1 2024-2025"
    ]
    SEMESTER_MAPPING = {sem: i for i, sem in enumerate(SEMESTER_ORDER)}

