from typing import List, Dict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from config.config_data import CONFIG_DATA

def process_data(df: pd.DataFrame, 
                 mapping_dict: Dict[str, str], 
                 target_cols: List[str] = CONFIG_DATA.TARGET_COLS) -> pd.DataFrame:
    """
    Tiền xử lý dữ liệu điểm thi THPTQG gồm thống nhất tên cột, xử lý điểm ngoại ngữ

    Tham số:
        df: DataFrame chứa dữ liệu điểm thô
        mapping_dict: Từ điển ánh xạ tên cột cũ sang tên cột mới chuẩn (VD: {"Toan": "MATH"})
        target_cols: Danh sách các cột bắt buộc phải có trong kết quả đầu ra

    Trả về:
        pd.DataFrame: DataFrame đã được xử lý và chỉ chứa các cột target_cols
    """

    df = df.rename(columns=mapping_dict)

    if "english" in df.columns:
        lang_cols = ["english", "russian", "french", "chinese", "german", "japanese"]
        lang_codes = ["N1", "N2", "N3", "N4", "N5", "N6"]
        
        valid_lang_cols = [c for c in lang_cols if c in df.columns]
        if valid_lang_cols:
            # Lấy điểm cao nhất trong các môn ngoại ngữ làm điểm FOREIGN_LANGUAGE
            df["FOREIGN_LANGUAGE"] = df[valid_lang_cols].max(axis=1)

            # Xác định mã ngoại ngữ (FOREIGN_LANGUAGE_CODE) dựa trên môn có điểm
            conditions = [df[col].notna() for col in lang_cols]
            df["FOREIGN_LANGUAGE_CODE"] = np.select(conditions, lang_codes, default=None)

    # Đảm bảo cột mã ngoại ngữ tồn tại
    if "FOREIGN_LANGUAGE_CODE" not in df.columns:
        df["FOREIGN_LANGUAGE_CODE"] = np.nan

    # Chuyển đổi sang dạng object để xử lý string
    df["FOREIGN_LANGUAGE_CODE"] = df["FOREIGN_LANGUAGE_CODE"].astype("object")
        
    # Xử lý trường hợp có điểm ngoại ngữ nhưng mất mã => Mặc định gán là N1 (Tiếng Anh)
    missing_code_mask = (df["FOREIGN_LANGUAGE"].notna()) & (df["FOREIGN_LANGUAGE_CODE"].isna())
    if missing_code_mask.any():
        df.loc[missing_code_mask, "FOREIGN_LANGUAGE_CODE"] = "N1"
        
    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan 
            
    return df[target_cols]


def get_information(df: pd.DataFrame, 
                    combo_list: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Tính toán thống kê (Mean, Std, Median) cho danh sách các tổ hợp xét tuyển.

    Tham số:
        df: DataFrame điểm thi đã được tiền xử lý
        combo_list: Danh sách các mã tổ hợp cần tính toán (VD: ["A00", "A01"]).

    Trả về:
        Dict[str, Dict[str, float]]: Kết quả thống kê dạng {"A00": {"mean": 8.5, ...}, ...}.
    """
    results = {}
    valid_scores_pool = [] 
    missing_combos_flag = False 

    for combo in combo_list:
        # Kiểm tra tổ hợp có trong công thức đã định nghĩa hay không
        if combo in CONFIG_DATA.SCORE_FORMULAS:
            config = CONFIG_DATA.SCORE_FORMULAS[combo]
            cols = config["cols"]
            lang_req = config["lang_req"]
            
            # Kiểm tra xem DataFrame có đủ các cột môn học cần thiết không
            if set(cols).issubset(df.columns):
                # Lọc theo yêu cầu ngoại ngữ (nếu có)
                if lang_req:
                    df_filtered = df[df["FOREIGN_LANGUAGE_CODE"] == lang_req]
                else:
                    df_filtered = df
                
                if not df_filtered.empty:
                    # Tính tổng điểm (chỉ tính khi đủ điểm tất cả các môn trong tổ hợp)
                    total_series = df_filtered[cols].sum(axis=1, min_count=len(cols))
                    total_series = total_series.dropna()
                    
                    if not total_series.empty:
                        stats = {
                            "mean": total_series.mean(),
                            "std": total_series.std(),
                            "median": total_series.median()
                        }
                        results[combo] = stats
                        valid_scores_pool.append(total_series)
                    else:
                        # Có cột nhưng tất cả dữ liệu đều NaN
                        missing_combos_flag = True
                else:
                    # Không có dòng dữ liệu nào thỏa mãn điều kiện lọc (VD: không có học sinh thi tiếng Pháp)
                    missing_combos_flag = True
            else:
                # Thiếu cột dữ liệu trong DataFrame đầu vào
                missing_combos_flag = True
        else:
            # Mã tổ hợp không tồn tại trong SCORE_FORMULAS
            missing_combos_flag = True

    # Coi tất cả tổ hợp không tính được là "OTHER" và tính thống kê chung
    if missing_combos_flag:
        if valid_scores_pool:
            # Gộp tất cả các điểm hợp lệ đã tính được từ các tổ hợp khác
            all_valid_scores = pd.concat(valid_scores_pool)
            results["OTHER"] = {
                "mean": all_valid_scores.mean(),
                "std": all_valid_scores.std(),
                "median": all_valid_scores.median()
            }
        else:
            results["OTHER"] = {"mean": 0, "std": 0, "median": 0}

    return results