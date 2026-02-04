import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

from utils import load_model, get_pred

def explain_model_shap(model_path: str, 
                       data_path: str, 
                       feats: List[str], 
                       target_col: str = "TC_HOANTHANH", 
                       top_n: int = 7,
                       approach_type: str = "Credits") -> Tuple[pd.DataFrame, shap.Explanation]:
    """
    Phân tích và giải thích mô hình sử dụng thư viện SHAP
    
    Hàm thực hiện 3 cấp độ phân tích:
    - Global: Tác động tổng thể của các features (Beeswarm plot)
    - Local: Giải thích chi tiết cho các mẫu đại diện (Best, Median, Worst prediction)
    - Contrastive: So sánh sự khác biệt giữa trường hợp dự đoán sai nhất và trường hợp tốt nhất gần nó

    Tham số:
        model_path: Đường dẫn đến file model đã lưu
        data_path: Đường dẫn đến file dữ liệu
        feats: Danh sách các features đầu vào
        target_col: Tên cột mục tiêu
        top_n: Số lượng feature hàng đầu muốn hiển thị trên biểu đồ
        approach_type: Phương pháp tiếp cận ("Credits", "Gap", "Ratio")

    Trả về:
        Tuple[pd.DataFrame, shap.Explanation]: 
            - DataFrame chứa kết quả dự đoán và sai số tuyệt đối
            - Đối tượng SHAP Explanation chứa giá trị SHAP của tập mẫu
    """
    df = pd.read_csv(data_path)
    X = df[feats]
    y_true = df[target_col].values

    model = load_model(model_path)

    if isinstance(model, lgb.Booster):
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X)
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
    
    y_pred = get_pred(y_pred, df["TC_DANGKY"].to_numpy(), approach_type)
    errors = np.abs(y_true - y_pred)
    df_res = df.copy()
    df_res["pred"] = y_pred
    df_res["abs_error"] = errors

    # Lấy top tốt nhất (lỗi thấp), top tệ nhất (lỗi cao) và nhóm giữa (median)
    n_total = len(df)
    n_sample = max(1, n_total // 10) # Lấy 10% cho mỗi nhóm
    idx_sorted = errors.argsort()
    
    selected_indices = np.concatenate([
        idx_sorted[:n_sample], # Index của nhóm sai số thấp nhất (Best cases)
        idx_sorted[(n_total // 2) - (n_sample // 2) : (n_total // 2) + (n_sample // 2)], # Index của nhóm sai số trung bình (Median cases)
        idx_sorted[-n_sample:] # Index của nhóm sai số cao nhất (Worst cases)
    ])
    
    X_subset = X.iloc[selected_indices]
    y_true_subset = y_true[selected_indices]
    
    try:
        # print("Attempting to use TreeExplainer...")
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        
        # Thử ép kiểu model_output="raw" để tránh lỗi parse base_score đôi khi gặp phải
        explainer = shap.TreeExplainer(booster, data=X_subset, model_output="raw")
        shap_values = explainer.shap_values(X_subset)
        expected_value = explainer.expected_value
        
    except Exception as e:
        # print(f"TreeExplainer failed ({e}). Falling back to KernelExplainer...")
        
        # Sử dụng KMeans để tóm tắt dữ liệu nền (giảm tải tính toán)
        # Convert sang numpy để tránh cảnh báo feature names
        background_data = shap.kmeans(X.values, 10) 
        
        if isinstance(model, lgb.Booster):
            predict_fn = lambda x: model.predict(x)
        else:
            predict_fn = lambda x: model.predict(x)

        explainer = shap.KernelExplainer(predict_fn, background_data)
        
        # KernelExplainer rất chậm, ta giới hạn nsamples
        shap_values = explainer.shap_values(X_subset, nsamples=100)
        expected_value = explainer.expected_value

    # KernelExplainer thường trả về list nếu là regression
    if isinstance(shap_values, list): 
        shap_values = shap_values[0]
    
    # Kiểm tra chiều dữ liệu
    if len(shap_values.shape) == 3: 
        shap_values = shap_values.mean(axis=2) 
        
    if isinstance(expected_value, (list, np.ndarray)):
        if len(np.atleast_1d(expected_value)) > 1:
            expected_value = expected_value[0]
        else:
            expected_value = float(expected_value)

    # Tạo đối tượng Explanation
    exp = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_subset.values,
        feature_names=feats
    )

    print(f"\n=== GLOBAL IMPACT ===")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(exp, max_display=top_n, show=False)
    plt.show()

    case_positions = {
        "BEST CASE (Min Error)": 0, # Đầu danh sách
        "MEDIAN CASE (Avg Error)": n_sample + (n_sample // 2), # Giữa danh sách
        "WORST CASE (Max Error)": len(selected_indices) - 1 # Cuối danh sách
    }

    print("\n=== REPRESENTATIVE SAMPLES ===")
    for label, pos in case_positions.items():
        real_idx = selected_indices[pos]
        mssv = df.iloc[real_idx]["MA_SO_SV"]
        print(f"\n{label} | MSSV: {mssv} | True: {y_true_subset[pos]:.2f} | Pred: {y_pred[selected_indices[pos]]:.2f}")
        plt.figure()
        shap.plots.waterfall(exp[pos], max_display=top_n, show=False)
        plt.title(f"{label} - MSSV: {mssv}")
        plt.show()
    

    print("\n=== CONTRASTIVE ANALYSIS ===")
    worst_idx_sub = len(selected_indices) - 1
    distances = np.linalg.norm(X_subset.iloc[:n_sample].values - X_subset.iloc[worst_idx_sub].values, axis=1)
    closest_best_pos = np.argmin(distances)
    
    mssv_worst = df.iloc[selected_indices[worst_idx_sub]]["MA_SO_SV"]
    mssv_best = df.iloc[selected_indices[closest_best_pos]]["MA_SO_SV"]
    
    diff_exp = shap.Explanation(
        values=shap_values[worst_idx_sub] - shap_values[closest_best_pos],
        base_values=0,
        data=X_subset.iloc[worst_idx_sub] - X_subset.iloc[closest_best_pos],
        feature_names=feats
    )
    
    plt.figure()
    shap.plots.bar(diff_exp, max_display=top_n, show=False)
    plt.title(f"Contrastive: {mssv_worst} vs {mssv_best}")
    plt.show()

    return df_res, exp