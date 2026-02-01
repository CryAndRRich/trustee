from typing import Dict, Any, List, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
import optuna

from config import CONFIG_MODEL
from utils import get_pred

def get_optuna_lgb_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        # Learning Control
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.01, 1.99),
        
        # Tree Structure
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 100, log=True),
        
        # Regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 10.0),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        
        # Sampling & Speed
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        "max_bin": trial.suggest_categorical("max_bin", [255, 512]),
    }
    
    return params

def optimize_lgb(train_df: pd.DataFrame, 
                 val_df: pd.DataFrame, 
                 feats: List[str], 
                 target_col: str, 
                 n_trial: int, 
                 model_type: str = "Fresher", 
                 approach_type: str = "Credits") -> Tuple[Dict[str, Any], float]:
    
    def objective_lgb(trial: optuna.Trial) -> float:
        params = {
            **CONFIG_MODEL.FIXED_PARAMS["LightGBM"],
            **get_optuna_lgb_params(trial)
        }
        
        model = LGBMRegressor(**params)
        callbacks = [
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0)
        ]
        
        model.fit(
            train_df[feats], 
            train_df[target_col],
            eval_set=[(val_df[feats], val_df[target_col])],
            eval_metric="rmse",
            callbacks=callbacks
        )
        best_iter = model.best_iteration_

        preds = model.predict(val_df[feats], num_iteration=best_iter)
        limit = val_df["TC_DANGKY"].to_numpy()

        final_preds = get_pred(preds, limit, approach_type)
        
        rmse = np.sqrt(mean_squared_error(val_df[target_col], final_preds))
        return rmse

    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=CONFIG_MODEL.RANDOM_SEED)
    )
    
    with tqdm(total=n_trial, desc=f"LightGBM {model_type} Tuning - Mode: {approach_type}") as pbar:
        def tqdm_callback(study):
            pbar.update(1)
            pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
    
        study.optimize(objective_lgb, n_trials=n_trial, callbacks=[tqdm_callback])

    best_params = {
        **CONFIG_MODEL.FIXED_PARAMS["LightGBM"], 
        **study.best_params
    }
    best_rmse = study.best_value
    
    return best_params, best_rmse