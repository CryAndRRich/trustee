from typing import Tuple, Union
import numpy as np

from .train import train_dt, train_rf, train_xgb, train_lgb

def train_model(model_name: str, 
                **kwargs) -> Tuple[Union[int, None], np.ndarray]:
    """
    Hàm gọi quá trình huấn luyện cho từng loại mô hình

    Tham số:
        model_name: Tên mô hình ("Decision Tree", "Random Forest", "XGBoost", "LightGBM")
        **kwargs: Các tham số truyền vào hàm train
            - params: Tham số mô hình
            - train_df: Tập dữ liệu huấn luyện
            - val_df: Tập dữ liệu kiểm định
            - feats: Danh sách các cột đặc trưng
            - target_cols: Cột mục tiêu
            - model_type: Tên loại mô hình ("Fresher", "Senior")
            - approach_type: Phương pháp tiếp cận ("Credits", "Gap", "Ratio")

    Trả về:
        Tuple[Union[int, None], np.ndarray]: (Best Iteration, Predictions)
    """
    
    train_map = {
        "Decision Tree": train_dt,
        "Random Forest": train_rf,
        "XGBoost": train_xgb,
        "LightGBM": train_lgb
    }

    # Kiểm tra tính hợp lệ của model name
    if model_name not in train_map:
        supported_models = ", ".join(train_map.keys())
        raise ValueError(
            f"Model '{model_name}' chưa được hỗ trợ. "
            f"Vui lòng chọn một trong các model sau: [{supported_models}]"
        )

    # Gọi hàm tương ứng
    train_func = train_map[model_name]
    return train_func(**kwargs)