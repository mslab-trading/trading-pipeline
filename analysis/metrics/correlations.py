import numpy as np

def daily_correlation(pred_df, truth_df):
    # 移除日期欄位，留下純數值
    if 'date' in pred_df.columns:
        pred_values = pred_df.drop(columns=['date']).copy()
        truth_values = truth_df.drop(columns=['date']).copy()
    else:
        pred_values = pred_df.copy()
        truth_values = truth_df.copy()

    # 每一天的相關係數（橫向）：計算每一天預測和真實之間的相關
    daily_correlations = []
    for i in range(len(pred_values)):
        pred_row = pred_values.iloc[i]
        truth_row = truth_values.iloc[i]
        # 計算當日不同股票之間的相關性
        corr = pred_row.corr(truth_row)
        daily_correlations.append(corr)
    
    avg_daily_corr = sum(daily_correlations) / len(daily_correlations)
    return avg_daily_corr

def stockly_correlation(pred_df, truth_df):
    stockly_corr = pred_df.corrwith(truth_df, axis=0).sort_values(ascending=False)
    return stockly_corr.mean()

def total_correlation(pred_df, truth_df):
    import numpy.ma as ma
    pred_values = pred_df.values.flatten()
    truth_values = truth_df.values.flatten()
    return ma.corrcoef(ma.masked_invalid(pred_values), ma.masked_invalid(truth_values))[0][1]
