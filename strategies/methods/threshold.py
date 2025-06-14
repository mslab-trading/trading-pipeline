import pandas as pd
def generate_signal_threshold(pred_df:pd.DataFrame, val_df:pd.DataFrame=None, threshold=0.5):
    signals = pred_df > threshold
    return signals