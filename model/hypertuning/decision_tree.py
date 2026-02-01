from typing import Dict, Any, List, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import optuna

from config import CONFIG_MODEL
from utils import get_pred

def get_optuna_dt_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        # Core & Criterion
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "poisson"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),

        # Structure Control
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 500, log=True),
        
        # Splitting & Leaf Constraints
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),

        # Pruning & Regularization
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.05),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.1),
    }
    
    return params

def optimize_dt(train_df: pd.DataFrame, 
                val_df: pd.DataFrame, 
                feats: List[str], 
                target_col: str, 
                n_trial: int, 
                model_type: str = "Fresher", 
                approach_type: str = "Credits") -> Tuple[Dict[str, Any], float]:
    
    def objective_dt(trial: optuna.Trial) -> float:
        params = {
            **CONFIG_MODEL.FIXED_PARAMS["Decision Tree"],
            **get_optuna_dt_params(trial)
        }
        
        model = DecisionTreeRegressor(**params)
        
        model.fit(
            train_df[feats], 
            train_df[target_col]
        )
        
        preds = model.predict(val_df[feats])
        limit = val_df["TC_DANGKY"].to_numpy()

        final_preds = get_pred(preds, limit, approach_type)
        
        rmse = np.sqrt(mean_squared_error(val_df[target_col], final_preds))
        return rmse

    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=CONFIG_MODEL.RANDOM_SEED)
    )
    
    with tqdm(total=n_trial, desc=f"Decision Tree {model_type} Tuning - Mode: {approach_type}") as pbar:
        def tqdm_callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
    
        study.optimize(objective_dt, n_trials=n_trial, callbacks=[tqdm_callback])

    best_params = {
        **CONFIG_MODEL.FIXED_PARAMS["Decision Tree"], 
        **study.best_params
    }
    best_rmse = study.best_value

    return best_params, best_rmse