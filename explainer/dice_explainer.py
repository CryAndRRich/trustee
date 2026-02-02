import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import dice_ml

from utils import load_model, get_pred


def explain_model_dice(model_path: str, 
                       train_path: str, 
                       val_path: str, 
                       feats: List[str], 
                       target_col: str = "TC_HOANTHANH", 
                       total_CFs: int = 5,
                       approach_type: str = "Credits") -> Any:
    """
    Tạo các phản ví dụ (Counterfactual Explanations) sử dụng thư viện DiCE.
    Giúp trả lời: "Cần thay đổi gì để đạt được kết quả mong muốn?"

    Tham số:
        model_path: Đường dẫn đến file model đã lưu
        train_path: Đường dẫn tập Train
        val_path: Đường dẫn tập Validation
        feats: Danh sách các features đầu vào
        target_col: Tên cột mục tiêu
        total_CFs: Số lượng phương án gợi ý muốn tạo ra
        desired_range: Khoảng giá trị mục tiêu mong muốn [min, max]
        approach_type: Phương pháp tiếp cận ("Credits", "Gap", "Ratio")

    Trả về:
        dice_exp: Đối tượng kết quả của DiCE
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    model = load_model(model_path)

    # Cấu hình DiCE Data
    train_dice_df = train_df[feats + [target_col]].copy()
    train_dice_df[target_col] = train_dice_df[target_col].astype(float)
    
    d = dice_ml.Data(dataframe=train_dice_df, 
                     continuous_features=feats, 
                     outcome_name=target_col)
    
    # Cấu hình DiCE Model
    m = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")

    exp = dice_ml.Dice(d, m, method="random")

    X_val = val_df[feats]
    if isinstance(model, lgb.Booster):
        y_pred = model.predict(X_val)
    else:
        y_pred = model.predict(X_val).flatten()
    
    y_pred = get_pred(y_pred, val_df["TC_DANGKY"].to_numpy(), approach_type)
    worst_idx = np.argmin(y_pred)
    mssv_worst = val_df.iloc[worst_idx]["MA_SO_SV"]
    
    print(f"\n=== DICE COUNTERFACTUAL EXPLANATION ===")
    print(f"Tư vấn cho MSSV: {mssv_worst} | Dự báo hiện tại: {y_pred[worst_idx]:.4f}")

    query_instance = X_val.iloc[worst_idx:worst_idx+1]
    
    # Sinh phản ví dụ (Counterfactuals)
    try:
        dice_exp = exp.generate_counterfactuals(
            query_instance, 
            total_CFs=total_CFs, 
            desired_range=[0.8, 1.0] 
        )
    except:
        dice_exp = exp.generate_counterfactuals(
            query_instance, 
            total_CFs=total_CFs, 
            desired_class="opposite"
        )

    # Hiển thị kết quả (chỉ hiện các cột có sự thay đổi)
    dice_exp.visualize_as_dataframe(show_only_changes=True)

    return dice_exp