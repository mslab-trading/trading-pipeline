import pandas as pd
import math
import numpy as np

def top_ratio_overlap_rate(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    ratio: float = 0.10,
) -> float:

    # Sanity checks
    if not (0 < ratio <= 1):
        raise ValueError("`ratio` must be in the open interval (0, 1].")
    if pred_df.shape != truth_df.shape:
        raise ValueError("`pred_df` and `truth_df` must have identical shape.")

    # Helper: flatten to a 1-D Series while preserving positional index
    def _flatten(df_like: pd.DataFrame) -> pd.Series:
        if isinstance(df_like, pd.DataFrame):
            return df_like.stack(dropna=False)          # keeps NaNs
        if isinstance(df_like, pd.Series):
            return df_like
        # Fallback for ndarray / list
        return pd.Series(np.asarray(df_like).ravel())

    pred = _flatten(pred_df)
    truth = _flatten(truth_df)
    n_items = len(pred)
    if n_items == 0:
        return float("nan")

    k = max(1, math.ceil(ratio * n_items))              # always keep ≥ 1

    top_pred_idx   = set(pred.nlargest(k).index)
    top_truth_idx  = set(truth.nlargest(k).index)
    intersection   = len(top_pred_idx & top_truth_idx)

    return intersection / k

def top_ratio_overlap_rate_daily(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    ratio: float = 0.10,
) -> float:
    """
    計算每日的 precision@ratio，並取平均。
    
    參數:
      pred: 預測的 DataFrame，必須包含 "date" 欄位，其餘欄位為各股票的預測值。
      truth: 真實值的 DataFrame，格式需與 pred 相同。
      ratio: 要挑選的比例 (預設為 0.1，即 10%)。
    
    邏輯:
      - 每日根據預測數值選出前 ratio 比例的股票集合 (top_pred)。
      - 每日根據真實數值選出前 ratio 比例的股票集合 (top_truth)。
      - 當天的 precision = |top_pred ∩ top_truth| / top_k，其中 top_k 為選取股票數量。
      - 返回所有日期 precision 的平均值。
    """
    precisions = []
    
    # 假設 pred 與 truth 皆有 "date" 欄位，並且兩個 DataFrame 的日期排列一致
    for i, pred_row in pred_df.iterrows():
        # 取得對應的真實值 row，假設 index 對應
        truth_row = truth_df.loc[i]
        
        # 排除 "date" 欄位，取得股票預測與真實值
        pred_stocks = pred_row.drop("date")
        truth_stocks = truth_row.drop("date")
        
        n_stocks = len(pred_stocks)
        top_k = max(1, int(ratio * n_stocks))  # 至少選 1 支股票
        
        # 依預測值排序，選出前 top_k 的股票名稱集合
        top_pred = set(pred_stocks.sort_values(ascending=False).head(top_k).index)
        # 依真實值排序，選出前 top_k 的股票名稱集合
        top_truth = set(truth_stocks.sort_values(ascending=False).head(top_k).index)
        
        # 當天 precision = 交集數量 / top_k
        daily_precision = len(top_pred.intersection(top_truth)) / top_k
        precisions.append(daily_precision)
        
    # 平均所有天數的 precision
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    return avg_precision

from typing import Callable

def precision_at_ratio(pred_df, truth_df, ratio, positive: Callable):
    total_elements = pred_df.size
    top_k_count = int(total_elements * ratio)
    
    # Flatten data
    pred_flat = pred_df.stack()
    truth_flat = truth_df.stack()
    
    # Sort by prediction descending
    top_k_pred = pred_flat.nlargest(top_k_count)
    
    # Calculate how many of the top predictions are actually positive
    correct = truth_flat[top_k_pred.index].apply(positive).sum()
    
    precision = correct / top_k_count
    return precision

def precision_daily_at_ratio(pred_df, truth_df, ratio, positive:Callable):
    daily_precision = []
    
    for date in pred_df.index:
        pred_row = pred_df.loc[date]
        truth_row = truth_df.loc[date]
        
        top_k_count = max(int(len(pred_row) * ratio), 1)
        
        top_k_pred = pred_row.nlargest(top_k_count)
        correct = truth_row[top_k_pred.index].apply(positive).sum()
        
        precision = correct / top_k_count
        daily_precision.append(precision)
    
    return np.mean(daily_precision)

def precision_baseline(truth_df, positive:Callable):
    total_elements = truth_df.size
    positive_count = truth_df.stack().apply(positive).sum()
    return positive_count / total_elements

def get_top_n_avg(pred_df: pd.DataFrame, truth_df: pd.DataFrame, n: int) -> float:
    """
    For each date in pred_df, pick the top-n stocks with highest predicted values,
    look up their true values in truth_df on the same date, and return the average
    of all those true values across all dates.
    
    Parameters:
    -----------
    pred_df : pd.DataFrame
        Prediction DataFrame indexed by date, columns = stock IDs, values = predicted scores.
    truth_df : pd.DataFrame
        Ground-truth DataFrame indexed by date, same columns = stock IDs, values = true returns.
    n : int
        Number of top stocks to pick each day (by predicted score).
    
    Returns:
    --------
    float
        The mean of the true values for the top-n predicted stocks over all dates.
    """
    true_values = []
    
    for date, pred_row in pred_df.iterrows():
        # Skip if this date isn't in the truth DataFrame
        if date not in truth_df.index:
            continue
        
        true_row = truth_df.loc[date]
        
        # Get the top-n stock IDs by prediction
        top_n_stocks = pred_row.nlargest(n).index
        
        # Collect their true values
        true_vals = true_row[top_n_stocks].values
        true_values.extend(true_vals)
    
    if len(true_values) == 0:
        return np.nan
    
    return np.mean(true_values)
    