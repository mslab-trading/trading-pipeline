import pandas as pd
import yaml
import os
from strategies.backtest.allen import *
from strategies.methods import allen_signal
from strategies.utils.analysis import print_result
from strategies.utils.data_processor import filter_bad_targets, get_price_df
from strategies.utils.analysis import get_equal_weight_baseline_result


def get_allen_signals(cfg: dict, result_dir: str, *, start_date=None, end_date=None):
    buy_dfs, sell_dfs = pd.DataFrame(), pd.DataFrame()
    for dir in os.listdir(result_dir):
        pred_df = pd.read_csv(os.path.join(result_dir, dir, "test/pred_pct.csv"), index_col="date")
        val_df = pd.read_csv(os.path.join(result_dir, dir, "train_val/pred_pct.csv"), index_col="date")
        buy_dfs  = pd.concat([ buy_dfs, allen_signal.get_buy_signals(pred_df, val_df, cfg)] , axis=0)
        sell_dfs = pd.concat([sell_dfs, allen_signal.get_sell_signals(pred_df, val_df, cfg)], axis=0)
    
    if start_date is not None:
        buy_dfs  = buy_dfs[ buy_dfs.index >= start_date.strftime("%Y-%m-%d")]
        sell_dfs = sell_dfs[sell_dfs.index >= start_date.strftime("%Y-%m-%d")]
    if end_date is not None:
        buy_dfs  = buy_dfs[ buy_dfs.index < end_date.strftime("%Y-%m-%d")]
        sell_dfs = sell_dfs[sell_dfs.index < end_date.strftime("%Y-%m-%d")]
    
    buy_dfs  =  buy_dfs.sort_index()
    sell_dfs = sell_dfs.sort_index()
    
    buy_dfs.index = pd.to_datetime(buy_dfs.index)
    sell_dfs.index = pd.to_datetime(sell_dfs.index)
    
    buy_dfs = buy_dfs.loc[~buy_dfs.index.duplicated(keep='last')]
    sell_dfs = sell_dfs.loc[~sell_dfs.index.duplicated(keep='last')]
    
    return {
        'buy_signals': buy_dfs,
        'sell_signals': sell_dfs,
    }
    
def get_allen_result(cfg: dict, result_dir: str, *, start_date=None, end_date=None):
    signals = get_allen_signals(cfg, result_dir, start_date=start_date, end_date=end_date)
    buy_dfs = signals['buy_signals']
    sell_dfs = signals['sell_signals']

    Target = filter_bad_targets(buy_dfs.columns, cfg)
        
    price_df = get_price_df(Target)
    price_df = price_df[(price_df.index >= buy_dfs.index[0]) & (price_df.index <= buy_dfs.index[-1])]
    
    backtest = Backtest(Strategy, price_df, commission=cfg["commission"], cash=1e9)
    result           = backtest.run(buy_dfs, sell_dfs, targets=Target, max_positions=len(Target), is_benchmark = False)
    result_benchmark = get_equal_weight_baseline_result(result_dir, buy_dfs.index[0], buy_dfs.index[-1])
    
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
