import numpy as np


def mse_loss(pred_values, truth_values):
    """
    計算均方誤差 (Mean Squared Error, MSE)
    
    參數:
      pred_values: 預測值 (numpy array)
      truth_values: 真實值 (numpy array)
    
    回傳:
      MSE 值
    """
    # 去除 NaN 值
    mask = ~np.isnan(pred_values) & ~np.isnan(truth_values)
    valid_pred = pred_values[mask]
    valid_truth = truth_values[mask]
    
    if len(valid_pred) == 0:
        return np.nan  # 或者返回 0，視需求而定
    
    mse_value = np.mean((valid_pred - valid_truth) ** 2)
    return mse_value
  