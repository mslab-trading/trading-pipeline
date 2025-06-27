from strategies.methods import allen_signal
from strategies.methods import berlin_signal
from strategies.methods import gino_signal
from strategies.methods import jerome_signal
import pandas as pd
import yaml

def generate_buy_signal(pred_df, method: str, threshold=0.):
    if (method == "gino"):
        return gino_signal.get_buy_signals(pred_df, threshold)
    raise NotImplementedError("This function is not implemented yet. Please use generate_sell_signal instead.")