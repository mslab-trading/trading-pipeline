from finlab import data
import finlab
import pandas as pd
import yaml
import os
from strategies.backtest.daily import *
from strategies.utils.data_processor import filter_bad_targets, get_price_df
from strategies.utils.analysis import get_equal_weight_baseline_result
from evaluation.stats import sharpe, sharpe_0050
import numpy as np

finlab_token = os.getenv("FINLAB_API_KEY")
finlab.login(finlab_token)


def get_daily_scores(cfg, pred_df: pd.DataFrame, val_df: pd.DataFrame):
    cash_pct = val_df.melt().value.quantile(cfg["cash_quantile"])
    pred_df["cash"] = cash_pct

    return pred_df


def compute_daily_signals(cfg, scores: pd.DataFrame):
    # Vectorized k-smoothed scores using EWM
    scores = scores.fillna(0)
    signals = scores.ewm(span=cfg["k"]).mean()

    # Vectorized softmax portfolio computation
    if cfg["normalize"]:
        signals = signals.sub(signals.mean(axis=1), axis=0).div(
            signals.std(axis=1), axis=0
        )

    signals = np.exp(signals / cfg["T"])
    signals = signals.div(signals.sum(axis=1), axis=0)

    # Vectorized cash thresholding
    cash_col = signals["cash"]
    signals = signals.where(signals.gt(cash_col, axis=0), 0.0)
    signals["cash"] = 1.0 - signals.drop(columns=["cash"]).sum(axis=1)

    return signals


def get_daily_signals(
    cfg: dict,
    result_dir: str,
    *,
    start_date=None,
    end_date=None,
):
    scores = pd.DataFrame()
    for dir in os.listdir(f"{result_dir}"):
        pred_df = pd.read_csv(f"{result_dir}/{dir}/test/pred_pct.csv", index_col=0)
        val_df = pd.read_csv(f"{result_dir}/{dir}/train_val/pred_pct.csv", index_col=0)
        _scores = get_daily_scores(cfg, pred_df, val_df)
        scores = pd.concat([scores, _scores], axis=0) if not scores.empty else _scores
    if start_date is not None:
        scores = scores[scores.index >= start_date.strftime("%Y-%m-%d")]
    if end_date is not None:
        scores = scores[scores.index < end_date.strftime("%Y-%m-%d")]
    scores = scores.sort_index()
    scores = scores.loc[~scores.index.duplicated(keep="last")]

    signals = compute_daily_signals(cfg, scores)

    return {
        "buy_signals": signals,
        "sell_signals": None
    }


def get_daily_result(
    cfg: dict,
    result_dir: str,
    *,
    start_date=None,
    end_date=None,
):
    signals = get_daily_signals(
        cfg,
        result_dir,
        start_date=start_date,
        end_date=end_date,
    )
    buy_signals = signals["buy_signals"]
    stocks = buy_signals.columns.tolist()
    stocks.remove("cash")
    Target = filter_bad_targets(stocks, cfg)

    price_df = get_price_df(Target)
    price_df = price_df[
        (price_df.index >= buy_signals.index[0])
        & (price_df.index <= buy_signals.index[-1])
    ]

    backtest = DailyBacktest(
        data=price_df,
        commission=cfg["commission"],
        tax=cfg["tax"],
        cash=1e9,
        freq="D",
    )
    result = backtest.run(buy_signal=buy_signals)
    benchmark_result = get_equal_weight_baseline_result(
        result_dir, buy_signals.index[0], buy_signals.index[-1]
    )

    return {
        "model": result,
        "benchmark": benchmark_result,
    }


def get_daily_metrics(
    cfg: dict,
    result_dir: str,
    start_date=None,
    end_date=None,
):
    """
    return:
    {
        'roi': float,
        'sharpe': float,
        'sharpe_0050': float,
    }
    """
    result = get_daily_result(
        cfg,
        result_dir,
        start_date=start_date,
        end_date=end_date,
    )

    roi = result["model"].returns.iloc[-1] / result["model"].returns.iloc[0] - 1
    sharpe_ratio = sharpe(result["model"].returns)

    # get 0050 close price
    with data.universe("ETF"):
        close_df = data.get("etl:adj_close")
        close_df = close_df["0050"]
        close_df = close_df[
            (close_df.index >= result["model"].returns.index[0])
            & (close_df.index <= result["model"].returns.index[-1])
        ]
    # etf_roi = close_df.iloc[-1] / close_df.iloc[0] - 1
    etf_sharpe = sharpe_0050(result["model"].returns, close_df)

    return {
        "roi": roi,
        "sharpe": sharpe_ratio,
        "sharpe_0050": etf_sharpe,
    }


if __name__ == "__main__":
    from strategies.run_allen import get_allen_result
    from strategies.run_gino import get_gino_result

    f = open("config/backtest.yaml")
    cfg = yaml.safe_load(f)
    result_dir = "results/Example_Result"

    start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-10-01", "%Y-%m-%d")

    buy_signals = get_daily_signals(
        cfg,
        result_dir,
        k=30,
        cash_quantile=0.5,
        start_date=start_date,
        end_date=end_date,
    )

    breakpoint()

    result = get_daily_result(
        cfg,
        result_dir,
        start_date=start_date,
        end_date=end_date,
    )
    # print(result)

    # breakpoint()
    roi = result["model"].returns.iloc[-1] / result["model"].returns.iloc[0] - 1
    print("ROI:", roi)
    sharpe_ratio = sharpe(result["model"].returns)
    print("Sharpe Ratio:", sharpe_ratio)

    breakpoint()

    with data.universe("ETF"):
        close_df = data.get("etl:adj_close")
        close_df = close_df["0050"]
    close_df = close_df[
        (close_df.index >= result["model"].returns.index[0])
        & (close_df.index <= result["model"].returns.index[-1])
    ]
    etf_roi = close_df.iloc[-1] / close_df.iloc[0] - 1
    print("0050 ROI:", etf_roi)
    etf_sharpe = sharpe_0050(result["model"].returns, close_df)
    print("Sharpe(0050) Ratio:", etf_sharpe)

    # result = get_gino_result(cfg, result_dir)
    # result_model = result['model']

    # print("If trading by our strategy:")
    # print_result(result_model.returns)
