from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta   # pip install python-dateutil


def read_all_df(
    result_dir: str | Path,
    start_year: int = 2021,
    end_year: int = 2025,
    *,
    horizon: str = "pct",        # "pct" 或 "abs" 等，避免魔術字串
    val_years: int = 1           # 驗證集長度（年）
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    讀取 result_dir 內所有 split 的 test / train_val csv，並依時間串接在一起。

    參數
    ----
    result_dir : str | Path
        儲存 result 的目錄
    start_year, end_year : int
        只保留 test 日期在 [start_year, end_year] 之間的資料
    horizon : str
        檔名的一部分，例如 pred_pct.csv / pred_abs.csv
    val_years : int
        要回溯多少「年」作為驗證集

    回傳
    ----
    all_pred      : DataFrame   # test 預測
    all_truth     : DataFrame   # test 真值
    all_pred_val  : DataFrame   # validation 預測
    all_truth_val : DataFrame   # validation 真值
    """
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise NotADirectoryError(result_dir)

    # -------------- 取得所有 split 子目錄 -----------------
    # 子目錄慣例：YYYYMMDD_YYYYMMDD 或 YYYY-MM-DD_YYYY-MM-DD
    pattern = re.compile(r"\d{4}-?\d{2}-?\d{2}_\d{4}-?\d{2}-?\d{2}")
    split_dirs = sorted(
        [p for p in result_dir.iterdir() if p.is_dir() and pattern.fullmatch(p.name)],
        key=lambda p: p.name
    )
    if not split_dirs:
        raise RuntimeError(f"No valid split folders under {result_dir}")

    # -------------- 預先建立空 DataFrame -----------------
    all_pred      = pd.DataFrame()
    all_truth     = pd.DataFrame()
    all_pred_val  = pd.DataFrame()
    all_truth_val = pd.DataFrame()

    # -------------- 依序讀取每個 split -------------------
    for split_path in split_dirs:
        # 解析起訖日期
        start_str, end_str = split_path.name.split("_")
        test_start = pd.to_datetime(start_str)
        test_end   = pd.to_datetime(end_str)

        # 若此 split 的 test 區間起始年不在範圍內，就略過
        if test_start.year < start_year or test_start.year > end_year:
            continue

        # validation 起點 = test_start - val_years
        val_start = test_start - relativedelta(years=val_years)

        # ------------ 檔案路徑 ------------
        test_dir  = split_path / "test"
        trv_dir   = split_path / "train_val"

        pred_file         = test_dir / f"pred_{horizon}.csv"
        truth_file        = test_dir / f"truth_{horizon}.csv"
        pred_train_val_fn = trv_dir / f"pred_{horizon}.csv"
        truth_train_val_fn= trv_dir / f"truth_{horizon}.csv"

        # ------------ 檢查檔案存在 ------------
        for fp in [pred_file, truth_file, pred_train_val_fn, truth_train_val_fn]:
            if not fp.is_file():
                raise FileNotFoundError(fp)

        # ------------ 讀取並標準化 ------------
        def _read(fp: Path) -> pd.DataFrame:
            df = pd.read_csv(fp)
            if "date" not in df.columns:
                raise ValueError(f"'date' column missing in {fp}")
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df.set_index("date").sort_index()
            return df

        pred_df         = _read(pred_file)
        truth_df        = _read(truth_file)
        pred_train_val  = _read(pred_train_val_fn)
        truth_train_val = _read(truth_train_val_fn)

        # ------------ 切出 validation 區段 ------------
        pred_val_df  = pred_train_val.loc[val_start:test_start - pd.Timedelta(days=1)]
        truth_val_df = truth_train_val.loc[val_start:test_start - pd.Timedelta(days=1)]

        # ------------ 串接至總表 ------------
        all_pred      = pd.concat([all_pred, pred_df])
        all_truth     = pd.concat([all_truth, truth_df])
        all_pred_val  = pd.concat([all_pred_val, pred_val_df])
        all_truth_val = pd.concat([all_truth_val, truth_val_df])

    # -------------- 去重 / 最後出現為準 ------------------
    # 若同一天 (index) 在不同 split 重複，使用最後一次結果
    all_pred      = all_pred.sort_index().groupby(level=0).last()
    all_truth     = all_truth.sort_index().groupby(level=0).last()
    all_pred_val  = all_pred_val.sort_index().groupby(level=0).last()
    all_truth_val = all_truth_val.sort_index().groupby(level=0).last()

    return all_pred, all_truth, all_pred_val, all_truth_val
