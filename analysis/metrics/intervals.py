import pandas as pd
import numpy as np

# ───────────────────────── group_intervals ─────────────────────────
def group_intervals(
    df: pd.DataFrame,
    stock: str,
    threshold: int,
    group_gap: int | pd.Timedelta = 10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    針對單一股票，將 (value > threshold) 的列合併成不重疊區間。

    ─ group_gap 兩種用法 ───────────────────────────────────────────
    • int        : 以「row index 的距離 ≤ group_gap」決定是否併入同一區間
    • Timedelta  : 以「時間差 ≤ group_gap」決定是否併入同一區間
      (若傳入其他非 int、非 Timedelta 的數值，會被視為天數轉成 Timedelta)

    回傳
    ----
    List[(start_ts, end_ts)]  — start_ts / end_ts 均為 pd.Timestamp
    """
    # 1) index 準備
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # 2) 所有高於 threshold 的 timestamp
    ts_list = df.index[df[stock] > threshold].tolist()
    if not ts_list:
        return []

    # 3) row-positions（僅在 row-gap 模式用得到）
    pos_list = [df.index.get_loc(ts) for ts in ts_list]

    # 4) 判斷使用 row-gap 或 time-gap
    if isinstance(group_gap, int):
        use_row_gap = True
        row_gap = group_gap
    else:
        use_row_gap = False
        if not isinstance(group_gap, pd.Timedelta):
            group_gap = pd.Timedelta(days=group_gap)

    # 5) 迭代合併
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i, T = 0, len(ts_list)
    while i < T:
        start_ts = prev_ts = ts_list[i]
        prev_pos = pos_list[i]
        j = i + 1

        while j < T:
            if use_row_gap:
                if pos_list[j] - prev_pos <= row_gap:
                    prev_ts, prev_pos, j = ts_list[j], pos_list[j], j + 1
                else:
                    break
            else:
                if ts_list[j] - prev_ts <= group_gap:
                    prev_ts, j = ts_list[j], j + 1
                else:
                    break

        intervals.append((start_ts, prev_ts))
        i = j

    return intervals


# ──────────────────────── intervals_overlap ────────────────────────
def intervals_overlap(interval1, interval2) -> bool:
    """
    檢查兩個 (Timestamp, Timestamp) 區間是否有交集。
    """
    start1, end1 = interval1
    start2, end2 = interval2
    return (start1 <= end2) and (start2 <= end1)


# ─────────────────────── extract_interval_matches ──────────────────
def extract_interval_matches(pred_df, truth_df, threshold, group_gap):
    """
    對每個股票：
      1. 以 group_intervals() 產生 pred / truth 區間
      2. 檢查每個 pred 區間是否與任一 truth 區間重疊

    回傳 dict 內容與舊版一致，但所有區間元素皆為 Timestamp。
    """
    # 確保 date 欄位為 Datetime
    if "date" in pred_df.columns:
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        truth_df["date"] = pd.to_datetime(truth_df["date"])

    result = {}
    for stock in pred_df.columns:
        if stock == "date":
            continue

        pred_intervals = group_intervals(pred_df, stock, threshold, group_gap)
        if not pred_intervals:  # 沒有預測區間直接跳過
            continue

        truth_intervals = group_intervals(truth_df, stock, threshold, group_gap)
        matched = [
            any(intervals_overlap(p, t) for t in truth_intervals)
            for p in pred_intervals
        ]

        result[stock] = {
            "pred_intervals": pred_intervals,
            "truth_intervals": truth_intervals,
            "matched_intervals": matched,
        }
    return result


# ───────────────────────── calculate_precision ─────────────────────
def calculate_precision(result) -> tuple[int, int, int]:
    """
    根據 extract_interval_matches 的輸出計算：
      • true_positive : 預測區間中與 truth 區間有交集者數量
      • total_positive: 預測區間總數
      • total_truth   : truth 區間總數
    （回傳 tuple 方便外部自行算 precision / recall）
    """
    true_positive = total_positive = total_truth = 0
    for stock in result.values():
        pred_intervals = stock["pred_intervals"]
        truth_intervals = stock["truth_intervals"]
        matched = stock["matched_intervals"]

        total_positive += len(pred_intervals)
        true_positive += sum(matched)
        total_truth += len(truth_intervals)

    return true_positive, total_positive, total_truth


# ──────────────────────── calculate_total_truth ────────────────────
def calculate_total_truth(truth_df, threshold, group_gap):
    """
    以 group_intervals() 對 truth_df 每檔股票計算 truth_intervals
    回傳 (truth_intervals_dict, total_truth_count)
    """
    truth_df = truth_df.copy()
    if "date" in truth_df.columns:
        truth_df["date"] = pd.to_datetime(truth_df["date"])

    truth_intervals_dict = {}
    total_truth = 0
    for stock in truth_df.columns:
        if stock == "date":
            continue
        intervals = group_intervals(truth_df, stock, threshold, group_gap)
        truth_intervals_dict[stock] = intervals
        total_truth += len(intervals)

    return truth_intervals_dict, total_truth

import numpy as np
import pandas as pd
from typing import Tuple

def get_prediction_interval_roi(
    pred_df: pd.DataFrame,
    df_close: pd.DataFrame,
    threshold: int,
    group_gap: int | pd.Timedelta,
    pred_len: int,
) -> Tuple[float, float]:
    """
    以預測訊號區間評估股票報酬率 (ROI)。

    Parameters
    ----------
    pred_df   : DataFrame
        每欄為一檔股票的預測分數，index 為日期。
    df_close  : DataFrame
        收盤價 (或其他參考價格)，欄位必須與 `pred_df` 相同。
    threshold : int
        判定「買進訊號」的門檻。> threshold 視為訊號成立。
    group_gap : int | Timedelta
        傳入 `group_intervals()`，用來合併相鄰訊號。
    pred_len  : int
        預期持有的最大長度 (row 數)。用於決定賣出觀察窗。

    Returns
    -------
    (mean_max_roi, mean_last_roi) : Tuple[float, float]
        • mean_max_roi  : 區間內「最佳賣出點」平均 ROI
        • mean_last_roi : 依照預設賣出日 (end_idx + pred_len) 平均 ROI
    """
    # ---------- 前置檢查與對齊 ----------
    # 統一日期型別
    for df in (pred_df, df_close):
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

    # 僅保留 pred_df 中存在於 df_close 的日期，以免 get_loc 失敗
    shared_idx = pred_df.index.intersection(df_close.index)
    pred_df = pred_df.loc[shared_idx]
    df_close = df_close.loc[shared_idx]

    max_rois: list[float] = []
    rois: list[float] = []

    for stock_id in (c for c in pred_df.columns if c != "date"):
        # 取得訊號區間 (Timestamp 形式)
        intervals = group_intervals(pred_df, stock_id, threshold, group_gap)
        if not intervals:
            continue

        price_series = df_close[stock_id]

        for start_ts, end_ts in intervals:
            # 若訊號區間超出收盤價範圍則略過
            if start_ts not in price_series.index or end_ts not in price_series.index:
                continue

            start_pos = price_series.index.get_loc(start_ts)
            end_pos   = price_series.index.get_loc(end_ts)
            sell_pos  = min(end_pos + pred_len, len(price_series) - 1)

            window = price_series.iloc[start_pos : sell_pos + 1]
            if window.empty or window.isna().all():
                continue

            start_price = window.iloc[0]
            max_price   = window.max()
            last_price  = window.iloc[-1]

            # 避免 0 或 NaN 問題
            if pd.isna(start_price) or start_price == 0:
                continue

            max_rois.append((max_price  - start_price) / start_price)
            rois.append    ((last_price - start_price) / start_price)

    # 若沒有任何有效區間，回傳 NaN
    if not max_rois:
        return float("nan"), float("nan")

    return float(np.mean(max_rois)), float(np.mean(rois))
