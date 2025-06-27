from finlab import data
import finlab
from pandas import MultiIndex
import pandas as pd
import json
import yaml
import os
import numpy as np
from strategies import generate_signal_topK
from strategies.backtest.pyramid import *
from strategies.utils.analysis import print_result
from strategies.utils.data_processor import filter_bad_targets, get_price_df
import math

def get_gino_signals(cfg: dict, result_dir: str):
    pred_df = None
    for dir in os.listdir(f'{result_dir}'):
        if pred_df is None:
            pred_df = pd.read_csv(f'{result_dir}/{dir}/test/pred_pct.csv', index_col=0)
        else:
            tmp = pd.read_csv(f'{result_dir}/{dir}/test/pred_pct.csv', index_col=0)
            pred_df = pd.concat([pred_df, tmp])
    pred_df = pred_df.sort_index()
    buy_signals = generate_signal_topK.generate_buy_signal(pred_df, 'gino')
    
    return {
        'buy_signals': buy_signals,
    }
    
def get_gino_result(cfg: dict, result_dir: str):
    signals = get_gino_signals(cfg, result_dir)
    buy_dfs = signals['buy_signals']

    Target = filter_bad_targets(list(buy_dfs.columns), cfg)
        
    finlab.login('ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m')
    price_df = get_price_df(Target)
    price_df = price_df[(price_df.index >= buy_dfs.index[0]) & (price_df.index <= buy_dfs.index[-1])]
    
    class MyStrategy(Strategy):
        def __init__(self, max_positions=40, max_holding_period=30, cash=1e9):
            super().__init__()
            self.max_positions = max_positions
            self.max_holding_period = max_holding_period
            self.cash = cash
        def positionSize(self, price: float):
            return math.floor((((self.cash + self.assets_value) / self.max_positions) / price) / (1 + 0.001425)) if price > 0 else 0
    
    backtest = PyramidBacktest(MyStrategy, price_df, commission=cfg["commission"], cash=1e9)
    result   = backtest.run(buy_signal=buy_dfs)
    return {
        'model': result,
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
    