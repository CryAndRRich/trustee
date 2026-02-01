import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lime import lime_tabular

from config import CONFIG_MODEL
from utils import load_model

def explain_model_lime(model_path: str, 
                       train_path: str,
                       val_path: str,
                       feats: List[str], 
                       target_col: str = "TC_HOANTHANH",
                       top_n: int = 7) -> np.ndarray:
    """
    Giải thích mô hình cục bộ sử dụng thư viện LIME. LIME tạo ra một mô hình tuyến tính 
    đơn giản xung quanh điểm dữ liệu cần giải thích để xấp xỉ hành vi của mô hình

    Tham số:
        model_path: Đường dẫn đến file model đã lưu
        train_path: Đường dẫn tập Train
        val_path: Đường dẫn tập Validation
        feats: Danh sách các features đầu vào
        target_col: Tên cột mục tiêu
        top_n: Số lượng feature hàng đầu muốn hiển thị trên biểu đồ

    Trả về:
        np.ndarray: Mảng chứa sai số tuyệt đối của tập validation
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    X_train = train_df[feats].values
    X_val = val_df[feats]
    y_true = val_df[target_col].values

    model = load_model(model_path)

    if isinstance(model, lgb.Booster):
        y_pred = model.predict(X_val.values)
    else:
        y_pred = model.predict(X_val.values).flatten()

    errors = np.abs(y_true - y_pred)
    
    idx_sorted = errors.argsort()
    n_total = len(val_df)
    case_indices = {
        "BEST CASE (Min Error)": idx_sorted[0],
        "MEDIAN CASE (Avg Error)": idx_sorted[n_total // 2],
        "WORST CASE (Max Error)": idx_sorted[-1]
    }

    # Khởi tạo LIME Explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feats,
        class_names=[target_col],
        mode="regression",
        verbose=False,
        random_state=CONFIG_MODEL.RANDOM_SEED
    )

    print("\n=== LIME LOCAL EXPLANATIONS ===")
    for label, idx in case_indices.items():
        mssv = val_df.iloc[idx]["MA_SO_SV"]
        print(f"\n{label} | MSSV: {mssv} | Error: {errors[idx]:.4f}")
        
        exp = explainer.explain_instance(
            data_row=X_val.values[idx], 
            predict_fn=predict_fn,
            num_features=top_n
        )
        
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME: {label} (MSSV: {mssv})")
        plt.tight_layout()
        plt.show()

    return errors