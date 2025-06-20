import strategies.methods.allen_signal
import strategies.methods.gino_signal
import pandas as pd

def generate_buy_signal(pred_df, method: str, val_df=None):
    raise NotImplementedError("This function is not implemented yet. Please use generate_sell_signal instead.")

def generate_sell_signal(pred_df:pd.DataFrame, method: str, val_df=None, buy_signal:pd.DataFrame=None):
    raise NotImplementedError("This function is not implemented yet. Please use generate_buy_signal instead.")
    