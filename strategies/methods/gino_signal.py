import pandas as pd

def get_buy_signals(pred_df: pd.DataFrame, threshold=0.):
    buy_signals = pred_df.apply(lambda x: x == x.max(), axis=1)
    buy_signals[pred_df < threshold] = False
    return buy_signals
