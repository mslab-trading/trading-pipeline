import pandas as pd
import yaml
import os
from strategies.methods import gino_signal
from strategies.backtest.pyramid import *
from strategies.utils.analysis import print_result
from strategies.utils.data_processor import filter_bad_targets, get_price_df
from strategies.utils.analysis import get_equal_weight_baseline_result
import math

def get_gino_signals(cfg: dict, result_dir: str, *, start_date=None, end_date=None):
    pred_df = None
    for dir in os.listdir(f'{result_dir}'):
        if pred_df is None:
            pred_df = pd.read_csv(f'{result_dir}/{dir}/test/pred_pct.csv', index_col=0)
        else:
            tmp = pd.read_csv(f'{result_dir}/{dir}/test/pred_pct.csv', index_col=0)
            pred_df = pd.concat([pred_df, tmp])
    pred_df = pred_df.groupby(['date']).tail(1)

    if start_date is not None:
        pred_df = pred_df[pred_df.index >= start_date.strftime("%Y-%m-%d")]
    if end_date is not None:
        pred_df = pred_df[pred_df.index < end_date.strftime("%Y-%m-%d")]
    
    pred_df = pred_df.sort_index()
    buy_signals = gino_signal.get_buy_signals(pred_df)
    
    return {
        'buy_signals': buy_signals,
        'sell_signals': pd.DataFrame(index=buy_signals.index, columns=buy_signals.columns, data=0),
    }

def get_gino_result(cfg: dict, result_dir: str, *, start_date=None, end_date=None):
    signals = get_gino_signals(cfg, result_dir, start_date=start_date, end_date=end_date)
    buy_dfs = signals['buy_signals']
    buy_dfs = buy_dfs.sort_index()

    Target = filter_bad_targets(list(buy_dfs.columns), cfg)
        
    price_df = get_price_df(Target)
    price_df = price_df[(price_df.index >= buy_dfs.index[0]) & (price_df.index <= buy_dfs.index[-1])]
    
    class MyStrategy(Strategy):
        def __init__(self, max_positions=cfg["max_positions"], max_holding_period=cfg["max_holding_period"], cash=1e9):
            super().__init__()
            self.max_positions = max_positions
            self.max_holding_period = max_holding_period
            self.cash = cash
        def positionSize(self, price: float):
            return math.floor((((self.cash + self.assets_value) / self.max_positions) / price) / (1 + 0.001425)) if price > 0 else 0
    
    backtest = PyramidBacktest(MyStrategy, price_df, commission=cfg["commission"], cash=1e9)
    result   = backtest.run(buy_signal=buy_dfs)
    benchmark_result = get_equal_weight_baseline_result(result_dir, buy_dfs.index[0], buy_dfs.index[-1])
    
    return {
        'model': result,
        'benchmark': benchmark_result
    }

if __name__ == "__main__":
    f = open("config/backtest.yaml")
    cfg = yaml.safe_load(f)
    result_dir = "results/Example_Result"
    # result_dir = f'/data2/Trading-Research/backtest_data_for_victor/results_add_series_decomp/Selected_0_ccc'
    
    result = get_gino_result(cfg, result_dir)
    result_model = result['model']

    print("If trading by our strategy:")
    print_result(result_model.returns)
    