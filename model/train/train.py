from typing import Any, Dict, List, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from utils import get_pred


def train_dt(params: Dict[str, Any], 
             train_df: pd.DataFrame, 
             val_df: pd.DataFrame, 
             feats: List[str], 
             target_cols: str, 
             model_type: str = "Fresher", 
             approach_type: str = "Credits") -> Tuple[None, np.ndarray]:
    print(f"Training Decision Tree {model_type} - Mode: {approach_type}...")
        
    model = DecisionTreeRegressor(**params)
    
    model.fit(
        train_df[feats], 
        train_df[target_cols]
    )
    
    preds = model.predict(val_df[feats])
    limit = val_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return None, final_preds


def train_rf(params: Dict[str, Any], 
             train_df: pd.DataFrame, 
             val_df: pd.DataFrame, 
             feats: List[str], 
             target_cols: str, 
             model_type: str = "Fresher", 
             approach_type: str = "Credits") -> Tuple[int, np.ndarray]:
    print(f"Training Random Forest {model_type} - Mode: {approach_type}...")
        
    model = RandomForestRegressor(**params)
    
    model.fit(
        train_df[feats], 
        train_df[target_cols]
    )
    
    best_iter = model.n_estimators
    preds = model.predict(val_df[feats])
    
    limit = val_df["TC_DANGKY"]
    limit = val_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return best_iter, final_preds


def train_xgb(params: Dict[str, Any], 
              train_df: pd.DataFrame, 
              val_df: pd.DataFrame, 
              feats: List[str], 
              target_cols: str, 
              model_type: str = "Fresher", 
              approach_type: str = "Credits") -> Tuple[int, np.ndarray]:
    print(f"Training XGBoost {model_type} - Mode: {approach_type}...")
        
    model = XGBRegressor(**params)
    model.fit(
        train_df[feats], 
        train_df[target_cols],
        eval_set=[(val_df[feats], val_df[target_cols])],
        verbose=100
    )
    
    best_iter = model.get_booster().best_iteration

    preds = model.predict(val_df[feats], iteration_range=(0, best_iter + 1))
    limit = val_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return best_iter, final_preds


def train_lgb(params: Dict[str, Any], 
              train_df: pd.DataFrame, 
              val_df: pd.DataFrame, 
              feats: List[str], 
              target_cols: str, 
              model_type: str = "Fresher", 
              approach_type: str = "Credits") -> Tuple[int, np.ndarray]:
    print(f"Training LightGBM {model_type} - Mode: {approach_type}...")
        
    model = LGBMRegressor(**params)
    callbacks = [
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100)
    ]
    
    model.fit(
        train_df[feats], 
        train_df[target_cols],
        eval_set=[(val_df[feats], val_df[target_cols])],
        eval_metric="rmse",
        callbacks=callbacks
    )
    
    best_iter = model.best_iteration_
    preds = model.predict(val_df[feats], num_iteration=best_iter)
    limit = val_df["TC_DANGKY"].to_numpy()
    final_preds = get_pred(preds, limit, approach_type)

    return best_iter, final_preds