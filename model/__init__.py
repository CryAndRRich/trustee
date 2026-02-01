import numpy as np

def get_pred(preds: np.ndarray, 
             limit: np.ndarray, 
             approach_type: str) -> np.ndarray:
    if approach_type == "Credits":
        final_preds = preds
    if approach_type == "Gap":
        final_preds = limit - preds
    elif approach_type == "Ratio":
        final_preds = limit * preds

    return np.clip(final_preds, 0, limit)

from .hypertuning import optimize_params
from .train import train_model
from .test import test_model

__all__ = [get_pred, optimize_params, train_model, test_model]