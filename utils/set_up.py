from typing import Any
import os
import random
import warnings
import joblib

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna

def set_seed(seed: int) -> None:
    """
    Cài đặt seed ngẫu nhiên để đảm bảo tính tái lập
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    
    warnings.filterwarnings("ignore", category=UserWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"Random seed set to {seed}")


def load_model(model_path: str) -> Any:
    print(f"Load model: {model_path.split("/")[-1]}")
    if model_path.endswith(".joblib"):
        model = joblib.load(model_path)
    elif model_path.endswith(".json"):
        model = xgb.XGBRegressor()
        model.load_model(model_path)
    elif model_path.endswith(".txt"):
        model = lgb.Booster(model_file=model_path)
    else:
        raise ValueError("Định dạng model không hỗ trợ (.joblib, .json, .txt)")

    return model