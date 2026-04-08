from __future__ import annotations

import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score


def regression_metrics(y_true, y_pred) -> dict:
    residuals = y_true - y_pred
    sd = np.std(residuals)
    bias = np.mean(residuals)
    lb = bias - 1.96 * sd
    ub = bias + 1.96 * sd

    tau, p_val = stats.kendalltau(0.5 * (y_true + y_pred), np.abs(residuals))
    mae = np.mean(np.abs(residuals))
    mae_sd = np.std(np.abs(residuals))
    rmse = np.sqrt((residuals ** 2).mean())
    mape = np.mean(np.abs(residuals / y_true)) * 100

    return {
        "bias": bias,
        "lb": lb,
        "ub": ub,
        "mae": mae,
        "mae_sd": mae_sd,
        "rmse": rmse,
        "precision": sd,
        "kendall_tau": tau,
        "kendall_p": p_val,
        "r2": r2_score(y_true, y_pred),
        "mape": mape,
    }


def print_metrics(model_name: str, metrics: dict) -> None:
    print(f"#{model_name}#")
    print("# -- Model -- #")
    print(f"Bias: {metrics['bias']:.3f} -- LB = {metrics['lb']:.3f} -- UB = {metrics['ub']:.3f}")
    print(f"MAE +/- SD = {metrics['mae']:.3f} +/- {metrics['mae_sd']:.3f}")
    print(f"RMSE = {metrics['rmse']:.3f}")
    print(f"Precision (residual std) = {metrics['precision']:.3f}")
    print(f"Kendall's tau = {metrics['kendall_tau']:.3f} (p = {metrics['kendall_p']:.4f})")
    print(f"R-Squared = {metrics['r2']:.3f}")
    print(f"MAPE = {metrics['mape']:.3f}%")
    print()
