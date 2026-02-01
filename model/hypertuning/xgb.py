from typing import Dict, Any, List, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna

from config import CONFIG_MODEL
from utils import get_pred


def get_optuna_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        # Learning Control
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.01, 1.99),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
        
        # Tree Structure
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512]),

        # Stochastic Sampling
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
    }
    
    if params["grow_policy"] == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 256)
        
    return params


def optimize_xgb(train_df: pd.DataFrame, 
                 val_df: pd.DataFrame, 
                 feats: List[str], 
                 target_col: str, 
                 n_trial: int, 
                 model_type: str = "Fresher", 
                 approach_type: str = "Credits") -> Tuple[Dict[str, Any], float]:
    
    base_score = train_df[target_col].mean()
    def objective_xgb(trial: optuna.Trial) -> float:
        params = {
            **CONFIG_MODEL.FIXED_PARAMS["XGBoost"],
            **get_optuna_xgb_params(trial),
            "base_score": base_score
        }
        
        model = XGBRegressor(**params)
        
        model.fit(
            train_df[feats], 
            train_df[target_col],
            eval_set=[(val_df[feats], val_df[target_col])],
            verbose=False
        )
        best_iter = model.get_booster().best_iteration

        preds = model.predict(val_df[feats], iteration_range=(0, best_iter + 1))
        limit = val_df["TC_DANGKY"].to_numpy()

        final_preds = get_pred(preds, limit, approach_type)
        
        rmse = np.sqrt(mean_squared_error(val_df["TC_HOANTHANH"], final_preds))
        return rmse

    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=CONFIG_MODEL.RANDOM_SEED)
    )
    
    with tqdm(total=n_trial, desc=f"XGBoost {model_type} Tuning - Mode: {approach_type}") as pbar:
        def tqdm_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
    
        study.optimize(objective_xgb, n_trials=n_trial, callbacks=[tqdm_callback])

    best_params = {
        **CONFIG_MODEL.FIXED_PARAMS["XGBoost"], 
        **study.best_params,
        "base_score": base_score
    }
    best_rmse = study.best_value
    
    return best_params, best_rmse