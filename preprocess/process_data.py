from typing import List, Tuple
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from config.config_data import CONFIG_DATA

def _calculate_z_score(row: pd.Series) -> float:
    """
    Hàm tính Z-Score cho từng sinh viên dựa trên năm và khối thi
    """
    year = row.get("NAM_TUYENSINH")
    score = row.get("DIEM_TRUNGTUYEN")

    # Chuẩn hóa tên tổ hợp (Upper case, xóa khoảng trắng)
    block = str(row.get("TOHOP_XT", "")).upper().strip()
    
    # Kiểm tra dữ liệu năm hợp lệ
    if year not in CONFIG_DATA.EXAM_STATS_DETAILED:
        return np.nan
        
    year_stats = CONFIG_DATA.EXAM_STATS_DETAILED[year]
    
    # Lấy thống kê của khối thi, nếu không có dùng "OTHER"
    stats = year_stats.get(block, year_stats["OTHER"])
        
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    # Tránh chia cho số quá nhỏ hoặc 0
    if std_val < 0.1: 
        std_val = 1.0
        
    return (score - mean_val) / std_val


def _parse_year(hk_str: str) -> int:
    """
    Trích xuất năm đang học từ chuỗi học kỳ (VD: "HK1 2023-2024" -> 2023)
    """
    hk_str = str(hk_str)
    year_match = re.search(r"(\d{4})", hk_str)
    return int(year_match.group(1)) if year_match else 2024


def get_data(admission_path: str, 
             academic_path: str, 
             test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Đọc dữ liệu, gộp bảng và chia tập Train/Val/Test

    Tham số:
        admission_path: Đường dẫn file thông tin tuyển sinh 
        academic_path: Đường dẫn file kết quả học tập
        test_path: Đường dẫn file danh sách sinh viên cần dự đoán

    Trả về:
        academic_df: Dữ liệu kết quả học tập
        student_df: Dữ liệu lịch sử sinh viên bao gồm thông tin tuyển sinh
        train_df_raw: Tập train thô
        val_df_raw: Tập validation thô (HK2 2023-2024)
        test_df_raw: Tập test thô
    """

    admission_df = pd.read_csv(admission_path)
    academic_df = pd.read_csv(academic_path)

    # Merge thông tin tuyển sinh vào lịch sử học tập
    student_df = pd.merge(academic_df, admission_df, on="MA_SO_SV", how="left")
    student_df["SEMESTER_INDEX"] = student_df["HOC_KY"].map(CONFIG_DATA.SEMESTER_MAPPING)

    # Sắp xếp và làm sạch dữ liệu trùng lặp
    student_df = student_df.sort_values(
        by=["MA_SO_SV", "HOC_KY", "TC_DANGKY", "TC_HOANTHANH", "GPA"],
        ascending=[True, True, True, False, False]
    )
    student_df = student_df.drop_duplicates(subset=["MA_SO_SV", "HOC_KY"], keep="first")

    # Sắp xếp lại theo thời gian để tính các feature dạng Time-series
    student_df = student_df.sort_values(["MA_SO_SV", "SEMESTER_INDEX"]).reset_index(drop=True)

    # Chia tách Train/Validation dựa trên học kỳ
    valid_semesters = ["HK2 2023-2024"]
    train_df_raw = student_df[~student_df["HOC_KY"].isin(valid_semesters)].copy()
    val_df_raw = student_df[student_df["HOC_KY"].isin(valid_semesters)].copy()

    # Xử lý tập Test
    test_df_raw = pd.read_csv(test_path)
    test_df_raw = pd.merge(test_df_raw, admission_df, on="MA_SO_SV", how="left")
    test_df_raw["SEMESTER_INDEX"] = test_df_raw["HOC_KY"].map(CONFIG_DATA.SEMESTER_MAPPING)

    # print(f"Train shape: {train_df_raw.shape}")
    # print(f"Val shape: {val_df_raw.shape}")
    # print(f"Test shape: {test_df_raw.shape}")

    return academic_df, student_df, train_df_raw, val_df_raw, test_df_raw

def get_features(input_df: pd.DataFrame, 
                 academic_df: pd.DataFrame, 
                 student_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng phức tạp:
    - Đặc trưng tuyển sinh (Z-score, Rank)
    - Đặc trưng chuỗi thời gian (Lag features, Rolling windows)
    - Đặc trưng tích lũy (Cumulative stats)

    Tham số:
        input_df: Dữ liệu sinh viên và học kỳ cần tính feature
        academic_df: Dữ liệu kết quả học tập
        student_df: Dữ liệu lịch sử toàn cục của sinh viên
    """
    df = input_df.copy()
    
    # Gộp dữ liệu quá khứ
    target_keys = set(zip(df["MA_SO_SV"], df["HOC_KY"]))
    student_filtered = student_df[
        ~student_df.set_index(["MA_SO_SV", "HOC_KY"]).index.isin(target_keys)
    ].copy()
    
    df = pd.concat([student_filtered, df], axis=0, ignore_index=True)
    df = df.sort_values(["MA_SO_SV", "SEMESTER_INDEX"])
        
    # Tính toán chênh lệch tín chỉ hoàn thành và đăng ký
    df["TC_HOANTHANH"] = df["TC_HOANTHANH"].astype(float)
    df["TC_DANGKY"] = df["TC_DANGKY"].astype(float)
    df["FAIL_CREDITS"] = df["TC_DANGKY"] - df["TC_HOANTHANH"]
    df["PASS_RATIO"] = df["TC_HOANTHANH"] / df["TC_DANGKY"]

    # Đặc trưng Tuyển sinh (Entrance Features)
    # Chuẩn hóa mã phương thức xét tuyển
    df["PTXT"] = df["PTXT"].replace({
        "5": "100",
        "3": "200",
        "1": "303"
    })

    # Tính Z-Score điểm đầu vào
    df["Z_SCORE"] = df.apply(_calculate_z_score, axis=1)

    # Tính toán chênh lệch điểm chuẩn và điểm trúng tuyển
    df["SCORE_GAP"] = df["DIEM_TRUNGTUYEN"] - df["DIEM_CHUAN"]
    df["GAP_RATIO"] = df["SCORE_GAP"] / (df["DIEM_CHUAN"] + 1.0)

    # Xếp hạng sinh viên trong cùng năm tuyển sinh
    df["ENTRY_RANK"] = df.groupby("NAM_TUYENSINH")["DIEM_TRUNGTUYEN"].transform(
        lambda x: x.rank(pct=True, method="average")
    )
    df["BENCHMARK_TIER"] = df.groupby("NAM_TUYENSINH")["DIEM_CHUAN"].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Đặc trưng Lịch sử học tập (Academic History Features)
    grouped = df.groupby("MA_SO_SV")

    # Tích lũy từ đầu đến hiện tại
    df["HIST_AVG_GPA"] = grouped["GPA"].transform(lambda x: x.shift(1).expanding().mean())
    df["TOTAL_EARNED"] = grouped["TC_HOANTHANH"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    df["HIST_MAX_PASSED"] = grouped["TC_HOANTHANH"].transform(lambda x: x.shift(1).expanding().max())
    df["HIST_MAX_GPA"] = grouped["GPA"].transform(lambda x: x.shift(1).expanding().max())
    df["HIST_STD_GPA"] = grouped["GPA"].transform(lambda x: x.shift(1).expanding().std())
    df["OVERLOAD_VS_MAX"] = df["TC_DANGKY"] - df["HIST_MAX_PASSED"] # So sánh đăng ký kỳ này với kỷ lục quá khứ

    # Kỳ liền trước
    df["LAST_GPA"] = grouped["GPA"].shift(1)
    df["LAST_FAIL"] = grouped["FAIL_CREDITS"].shift(1)
    df["LAST_PASSED"] = grouped["TC_HOANTHANH"].shift(1)
    df["LAST_DANGKY"] = grouped["TC_DANGKY"].shift(1)
    df["LAST_PASS_RATIO"] = df["LAST_PASSED"] / df["LAST_DANGKY"]

    # Hai kỳ gần nhất
    window2 = grouped.rolling(window=2, min_periods=1, closed="left")
    df["R2_AVG_GPA"] = window2["GPA"].mean().reset_index(0, drop=True)
    df["R2_SUM_FAIL"] = window2["FAIL_CREDITS"].sum().reset_index(0, drop=True)
    df["R2_AVG_PASSED"] = window2["TC_HOANTHANH"].mean().reset_index(0, drop=True)
    df["R2_SUM_DANGKY"] = window2["TC_DANGKY"].sum().reset_index(0, drop=True)
    df["R2_SUM_PASSED"] = window2["TC_HOANTHANH"].sum().reset_index(0, drop=True)
    df["R2_PASS_RATE"] = df["R2_SUM_PASSED"] / df["R2_SUM_DANGKY"]

    df["PRESSURE_VS_R2"] = df["TC_DANGKY"] / df["R2_AVG_PASSED"] # Áp lực so với trung bình gần đây
    df["GPA_TREND_R2"] = df["R2_AVG_GPA"] - df["HIST_AVG_GPA"] # Xu hướng điểm số (gần đây vs tích lũy)
    df["FAIL_TREND_R2"] = df["LAST_FAIL"] - (df["R2_SUM_FAIL"] / 2) # Xu hướng rớt môn

    # Ba kỳ gần nhất
    window3 = grouped.rolling(window=3, min_periods=1, closed="left")
    df["R3_AVG_GPA"] = window3["GPA"].mean().reset_index(0, drop=True)
    df["R3_SUM_FAIL"] = window3["FAIL_CREDITS"].sum().reset_index(0, drop=True)
    df["R3_AVG_PASSED"] = window3["TC_HOANTHANH"].mean().reset_index(0, drop=True)

    df["PRESSURE_VS_R3"] = df["TC_DANGKY"] / df["R3_AVG_PASSED"]
    df["OVERLOAD_R3"] = df["TC_DANGKY"] - df["R3_AVG_PASSED"]

    # Đặc trưng Niên khóa (School Year Features)
    df["YEAR_START"] = df["HOC_KY"].apply(_parse_year)
    df["SV_NAM_THU"] = df["YEAR_START"] - df["NAM_TUYENSINH"] + 1
    df.loc[df["SV_NAM_THU"] < 1, "SV_NAM_THU"] = 1 
    
    # Xử lý giá trị thiếu
    fill_0 = [
        "LAST_FAIL", "R2_SUM_FAIL", "R3_SUM_FAIL", "FAIL_TREND_R2", "OVERLOAD_R3", 
        "TOTAL_EARNED", "OVERLOAD_VS_MAX", "HIST_STD_GPA", "GPA_TREND_R2"
    ]
    df[fill_0] = df[fill_0].fillna(0.0)

    fill_1 = ["LAST_PASS_RATIO", "R2_PASS_RATE", "PRESSURE_VS_R2", "PRESSURE_VS_R3"]
    df[fill_1] = df[fill_1].fillna(1.0)
    
    fill_15 = ["LAST_PASSED", "R2_AVG_PASSED", "R3_AVG_PASSED", "HIST_MAX_PASSED"]
    df[fill_15] = df[fill_15].fillna(15.0)

    fill_mean_gpa = ["LAST_GPA", "R2_AVG_GPA", "R3_AVG_GPA", "HIST_AVG_GPA", "HIST_MAX_GPA"]
    df[fill_mean_gpa] = df[fill_mean_gpa].fillna(academic_df["GPA"].mean())
        
    # Xử lý vô cực và NaN còn sót lại
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Merge kết quả về lại DataFrame gốc
    final_df = pd.merge(
        input_df[["MA_SO_SV", "HOC_KY"]], 
        df, 
        on=["MA_SO_SV", "HOC_KY"], 
        how="left"
    )
    
    return final_df


def split_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tách dữ liệu thành sinh viên năm nhất và sinh viên các năm sau
    """
    df_fresh = df[df["SV_NAM_THU"] == 1].copy()
    df_senior = df[df["SV_NAM_THU"] > 1].copy()
    
    return df_fresh, df_senior


def filter_cols(df: pd.DataFrame, 
                features: List[str], 
                meta_cols: List[str]) -> pd.DataFrame:
    """
    Lọc lấy các cột cần thiết cho mode
    """
    desired_cols = set(features + meta_cols)
    existing_cols = [c for c in df.columns if c in desired_cols]
    return df[existing_cols].copy()