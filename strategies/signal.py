from strategies.methods.topk import generate_signal_topk
from strategies.methods.threshold import generate_signal_threshold  # 可選

def generate_buy_signal(pred_df, method: str, val_df=None):
    if method == "topk":
        return generate_signal_topk(pred_df, val_df)
    elif method == "threshold":
        return generate_signal_threshold(pred_df, val_df)
    else:
        raise ValueError(f"Unknown method: {method}")
    