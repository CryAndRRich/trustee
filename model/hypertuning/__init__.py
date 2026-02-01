from typing import Dict, Any, Tuple

from .decision_tree import optimize_dt
from .random_forest import optimize_rf
from .xgb import optimize_xgb
from .lgbm import optimize_lgb


def optimize_params(model_name: str, **kwargs) -> Tuple[Dict[str, Any], float]:
    """
    Hàm gọi quy trình tối ưu hóa cho từng loại model
    
    Tham số:
        model_name: Tên mô hình ("Decision Tree", "Random Forest", "XGBoost", "LightGBM")
        **kwargs: Các tham số truyền vào hàm optimize
            - train_df: Tập dữ liệu huấn luyện
            - val_df: Tập dữ liệu kiểm định
            - feats: Danh sách các cột đặc trưng
            - target_col: Tên cột mục tiêu cần dự đoán
            - n_trial: Số lần thử nghiệm của Optuna.
            - model_type: Tên loại mô hình ("Fresher", "Senior")
            - approach_type: Phương pháp tiếp cận ("Credits", "Gap", "Ratio")
        
    Trả về:
        Tuple[Dict[str, Any], float]: (Best Params, Best RMSE)
    """
    # Tạo từ điển ánh xạ
    optimizers = {
        "Decision Tree": optimize_dt,
        "Random Forest": optimize_rf,
        "XGBoost": optimize_xgb,
        "LightGBM": optimize_lgb
    }

    # Kiểm tra tính hợp lệ của model name
    if model_name not in optimizers:
        supported_models = ", ".join(optimizers.keys())
        raise ValueError(
            f"Model '{model_name}' chưa được hỗ trợ. "
            f"Vui lòng chọn một trong các model sau: [{supported_models}]"
        )

    # Gọi hàm tương ứng
    optimizer_func = optimizers[model_name]
    return optimizer_func(**kwargs)