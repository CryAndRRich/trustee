from typing import Union, List

import numpy as np
import pandas as pd

def save_predictions(test_fresh: pd.DataFrame,
                     test_senior: pd.DataFrame,
                     preds_fresh: Union[np.ndarray, List[float]],
                     preds_senior: Union[np.ndarray, List[float]],
                     test_raw: pd.DataFrame,
                     submission_path: str) -> pd.DataFrame:
    """
    Tổng hợp kết quả dự đoán, gộp lại và lưu file submission. Hàm đảm bảo thứ tự dòng 
    của file kết quả khớp 100% với file test_raw ban đầu

    Tham số:
        test_fresh: Tập dữ liệu test nhóm sinh viên năm nhất
        test_senior: Tập dữ liệu test nhóm sinh viên năm hai trở lên
        preds_fresh: Kết quả dự đoán tương ứng của nhóm Fresh
        preds_senior: Kết quả dự đoán tương ứng của nhóm Senior
        test_raw: File test gốc
        submission_path: Đường dẫn lưu file CSV kết quả

    Trả về:
        pd.DataFrame: DataFrame kết quả cuối cùng đã lưu
    """
    
    # Tạo DataFrame dự đoán cho từng nhóm
    sub_fresh_df = pd.DataFrame({
        "MA_SO_SV": test_fresh["MA_SO_SV"],
        "PRED_TC_HOANTHANH": preds_fresh
    })

    sub_senior_df = pd.DataFrame({
        "MA_SO_SV": test_senior["MA_SO_SV"],
        "PRED_TC_HOANTHANH": preds_senior
    })

    # Gộp hai nhóm lại thành một bảng duy nhất
    preds_combined = pd.concat([sub_fresh_df, sub_senior_df], axis=0, ignore_index=True)

    # Merge với file gốc để đảm bảo ĐÚNG THỨ TỰ và ĐỦ SỐ LƯỢNG sinh viên
    submission_df = pd.merge(
        test_raw[["MA_SO_SV"]],
        preds_combined,
        on="MA_SO_SV",
        how="left"
    )

    # Xử lý trường hợp bị khuyết (nếu có lỗi logic khiến ID không khớp)
    submission_df["PRED_TC_HOANTHANH"] = submission_df["PRED_TC_HOANTHANH"].fillna(0)

    submission_df.to_csv(submission_path, index=False)
    print(f"Saved Submission: {submission_path}")
    print(f"Shape: {submission_df.shape}")
    
    return submission_df