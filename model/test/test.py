from typing import Any, Dict, List
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from model import get_pred


def _calculate_scaled_estimators(best_iter: int, 
                                 full_train_len: int, 
                                 train_len: int) -> int:
    """
    Tính toán lại số lượng cây (estimators), tăng số lượng cây theo tỷ lệ kích thước dữ liệu tăng thêm
    """
    if train_len == 0: 
        return best_iter
    scale_ratio = full_train_len / train_len
    return int(best_iter * scale_ratio)


def _save_model(model: Any, 
                model_name: str, 
                save_dir: str,
                ext: str) -> None:
    """
    Lưu model vào thư mục
    """
    filename = f"{model_name}.{ext}"
    save_path = os.path.join(save_dir, filename)
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, "save_model"):
        # Dành cho XGBoost/LightGBM (native save)
        model.save_model(save_path)
    else:
        # Dành cho Sklearn (joblib)
        joblib.dump(model, save_path)
    
    print(f"Saved Model: {save_path}")


def test_dt(params: Dict[str, Any], 
            full_train_df: pd.DataFrame, 
            train_df: pd.DataFrame, 
            test_df: pd.DataFrame, 
            feats: List[str], 
            target_col: str,
            model_type: str = "Fresher",
            approach_type: str = "Credits") -> np.ndarray:
    print(f"Testing Decision Tree {model_type} - Mode: {approach_type}...")

    final_params = params.copy()
    model = DecisionTreeRegressor(**final_params)
    model.fit(
        full_train_df[feats], 
        full_train_df[target_col]
    )
    
    _save_model(model, f"decision_tree_{model_type.lower()}", "joblib")
    
    preds = model.predict(test_df[feats])
    limit = test_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return final_preds


def test_rf(params: Dict[str, Any], 
            best_iter: int,
            full_train_df: pd.DataFrame, 
            train_df: pd.DataFrame, 
            test_df: pd.DataFrame, 
            feats: List[str], 
            target_col: str,
            model_type: str = "Fresher",
            approach_type: str = "Credits") -> np.ndarray:
    print(f"Testing Random Forest {model_type} - Mode: {approach_type}...")

    final_params = params.copy()
    final_params["n_estimators"] = _calculate_scaled_estimators(
        best_iter, len(full_train_df), len(train_df)
    )

    model = RandomForestRegressor(**final_params)
    model.fit(
        full_train_df[feats], 
        full_train_df[target_col]
    )
    
    _save_model(model, f"random_forest_{model_type.lower()}", "joblib")
    
    preds = model.predict(test_df[feats])
    limit = test_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return final_preds


def test_xgb(params: Dict[str, Any], 
             best_iter: int,
             full_train_df: pd.DataFrame, 
             train_df: pd.DataFrame, 
             test_df: pd.DataFrame, 
             feats: List[str], 
             target_col: str,
             model_type: str,
             approach_type: str = "Credits") -> np.ndarray:
    print(f"Testing XGBoost {model_type} - Mode: {approach_type}...")

    final_params = params.copy()
    final_params["n_estimators"] = _calculate_scaled_estimators(
        best_iter, len(full_train_df), len(train_df)
    )
    if "early_stopping_rounds" in final_params:
        del final_params["early_stopping_rounds"]

    model = XGBRegressor(**final_params)
    model.fit(
        full_train_df[feats], 
        full_train_df[target_col],
        verbose=0
    )

    _save_model(model, f"xgboost_{model_type.lower()}", "json")
    
    preds = model.predict(test_df[feats])
    limit = test_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return final_preds


def test_lgb(params: Dict[str, Any], 
             best_iter: int,
             full_train_df: pd.DataFrame, 
             train_df: pd.DataFrame, 
             test_df: pd.DataFrame, 
             feats: List[str], 
             target_col: str,
             model_type: str,
             approach_type: str = "Credits") -> np.ndarray:
    print(f"Testing LightGBM {model_type} - Mode: {approach_type}...")

    final_params = params.copy()
    final_params["n_estimators"] = _calculate_scaled_estimators(
        best_iter, len(full_train_df), len(train_df)
    )
    if "early_stopping_rounds" in final_params:
        del final_params["early_stopping_rounds"]

    model = LGBMRegressor(**final_params)
    model.fit(
        full_train_df[feats], 
        full_train_df[target_col],
        eval_metric="rmse"
    )
    
    _save_model(model.booster_, f"lgbm_{model_type.lower()}", "txt")
    
    preds = model.predict(test_df[feats])
    limit = test_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return final_preds