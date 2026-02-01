import numpy as np

from .test import test_dt, test_rf, test_xgb, test_lgb

def test_model(model_name: str, 
               **kwargs) -> np.ndarray:
    """
    Hàm quá trình test (retrain full + predict) cho từng loại mô hình

    Tham số:
        model_name: Tên mô hình ("Decision Tree", "Random Forest", "XGBoost", "LightGBM")
        **kwargs: Các tham số truyền vào hàm train
            - params: Tham số mô hình
            - full_train_df: Tập dữ liệu huấn luyện đầy đủ
            - train_df: Tập dữ liệu huấn luyện ban đầu
            - test_df: Tập dữ liệu kiểm tra
            - feats: Danh sách các cột đặc trưng
            - target_cols: Cột mục tiêu
            - model_type: Tên loại mô hình ("Fresher", "Senior")
            - approach_type: Phương pháp tiếp cận ("Credits", "Gap", "Ratio")

    Trả về:
        np.ndarray: Mảng kết quả dự đoán
    """
    
    test_map = {
        "Decision Tree": test_dt,
        "Random Forest": test_rf,
        "XGBoost": test_xgb,
        "LightGBM": test_lgb
    }

    # Kiểm tra tính hợp lệ của model name
    if model_name not in test_map:
        supported_models = ", ".join(test_map.keys())
        raise ValueError(
            f"Model '{model_name}' chưa được hỗ trợ. "
            f"Vui lòng chọn một trong các model sau: [{supported_models}]"
        )

    # Gọi hàm tương ứng
    test_func = test_map[model_name]
    return test_func(**kwargs)