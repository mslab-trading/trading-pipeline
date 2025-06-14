def generate_signal_topk(pred_df, val_df=None):
    # 假設選每天預測前 5 檔
    topk = 5
    signals = pred_df.apply(lambda row: row.nlargest(topk).index.tolist(), axis=1)
    return signals