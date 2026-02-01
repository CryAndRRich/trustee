from typing import Union, List, Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def _calculate_wmape(y_true: np.ndarray, 
                    y_pred: np.ndarray) -> float:
    """
    Tính toán chỉ số wMAPE (Weighted Mean Absolute Percentage Error)
    """
    if np.sum(y_true) == 0: 
        return 0.0
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def evaluate_model_performance(y_true: Union[np.ndarray, List], 
                               y_pred: Union[np.ndarray, List], 
                               phase_name: str = "Validation") -> Dict[str, float]:
    """
    Tính toán và in ra các chỉ số đánh giá mô hình hồi quy

    Tham số:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán từ mô hình
        phase_name: Tên giai đoạn (VD: "Train", "Validation", "Test")

    Returns:
        Dict[str, float]: Từ điển chứa các giá trị {"RMSE", "MSE", "R2", "wMAPE"}
    """
    # Chuyển đổi sang numpy array để tính toán vector hóa
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tính toán các chỉ số
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    wmape = _calculate_wmape(y_true, y_pred)
    
    # In kết quả ra màn hình (Formatted Print)
    print(f"=== Performance Metrics [{phase_name} Set] ===")
    print(f"RMSE  : {rmse:.4f}")
    print(f"MSE   : {mse:.4f}")
    print(f"R^2   : {r2:.4f}")
    print(f"wMAPE : {wmape:.4f}")
    print("=" * 30)
    
    # Trả về kết quả dưới dạng dictionary
    metrics = {
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "wmape": wmape
    }
    
    return metrics