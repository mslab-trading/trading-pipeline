from strategies.methods import allen_signal
from strategies.methods import berlin_signal
from strategies.methods import gino_signal
from strategies.methods import jerome_signal
import pandas as pd
import yaml

def generate_buy_signal(pred_df, method: str, val_df=None):
    if (method == "allen"):
        f = open("config/backtest.yaml")
        cfg_dict_backtest = yaml.safe_load(f)
        f = open("config/main_training.yaml")
        cfg_dict_train = yaml.safe_load(f)
        cfg = cfg_dict_backtest | cfg_dict_train
        
        # These three can be modified
        cfg["ADX_threshold"] = 40
        cfg["buy_percentile"] = 0.75
        cfg["sell_percentile"] = 0.25
        return allen_signal.get_buy_signals(pred_df, val_df, cfg)
    raise NotImplementedError("This function is not implemented yet. Please use generate_sell_signal instead.")

def generate_sell_signal(pred_df:pd.DataFrame, method: str, val_df=None, buy_signal:pd.DataFrame=None):
    if (method == "allen"):
        f = open("config/backtest.yaml")
        cfg_dict_backtest = yaml.safe_load(f)
        f = open("config/main_training.yaml")
        cfg_dict_train = yaml.safe_load(f)
        cfg = cfg_dict_backtest | cfg_dict_train
        
        # These three can be modified
        cfg["ADX_threshold"] = 40
        cfg["buy_percentile"] = 0.75
        cfg["sell_percentile"] = 0.25
        return allen_signal.get_sell_signals(pred_df, val_df, cfg)
    raise NotImplementedError("This function is not implemented yet. Please use generate_buy_signal instead.")
    