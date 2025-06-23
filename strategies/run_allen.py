from finlab import data
import finlab
from pandas import MultiIndex
import pandas as pd
import json
import yaml
import os
import numpy as np
from strategies import generate_signal
from strategies.backtest.allen import *
from strategies.utils.analysis import print_result
from strategies.utils.data_processor import filter_bad_targets, get_price_df

def get_allen_result(cfg: dict, result_dir: str):
    buy_dfs, sell_dfs = pd.DataFrame(), pd.DataFrame()
    for dir in os.listdir(result_dir):
        pred_df = pd.read_csv(os.path.join(result_dir, dir, "test/pred_pct.csv"), index_col="date")
        val_df = pd.read_csv(os.path.join(result_dir, dir, "train_val/pred_pct.csv"), index_col="date")
        buy_dfs  = pd.concat([ buy_dfs, generate_signal.generate_buy_signal(pred_df, "allen", val_df)] , axis=0)
        sell_dfs = pd.concat([sell_dfs, generate_signal.generate_sell_signal(pred_df, "allen", val_df)], axis=0)
    buy_dfs  =  buy_dfs.sort_index()
    sell_dfs = sell_dfs.sort_index()

    Target = filter_bad_targets(buy_dfs.columns, cfg)
        
    finlab.login('ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m')
    price_df = get_price_df(Target)
    price_df = price_df[(price_df.index >= buy_dfs.index[0]) & (price_df.index <= buy_dfs.index[-1])]
    
    backtest = Backtest(Strategy, price_df, commission=cfg["commission"], cash=1e9)
    result           = backtest.run(buy_dfs, sell_dfs, targets=Target, max_positions=len(Target), is_benchmark = False)
    result_benchmark = backtest.run(buy_dfs, sell_dfs, targets=Target, max_positions=len(Target), is_benchmark = True)
    return {
        'model': result,
        'benchmark': result_benchmark,
    }

if __name__ == "__main__":
    f = open("config/backtest.yaml")
    cfg = yaml.safe_load(f)
    result_dir = "results/Example_Result"
    
    result = get_allen_result(cfg, result_dir)
    result_benchmark = result['benchmark']
    result_model = result['model']

    print("If trading by our strategy:")
    print_result(result_model.returns)
    print("If trading by benchmark:")
    print_result(result_benchmark.returns)
