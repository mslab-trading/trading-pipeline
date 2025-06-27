import pandas as pd

def get_buy_signals(pred_df, threshold=0.):
    datetime_list = pred_df.index.tolist()
    stock_names = list(pred_df.columns)
    
    signal = {}
    for name in stock_names:
        signal[f'{name}'] = [0. for _ in range(len(datetime_list))]
    
    start_date = pred_df.index[0]
    end_date = pred_df.index[-1]
    start_idx = datetime_list.index(start_date)
    end_idx = datetime_list.index(end_date)
    
    def get_signal():

        idx = start_idx
        while idx <= end_idx:
            date = datetime_list[idx]
            if date in pred_df.index:
                stock_id = pred_df.loc[date].idxmax()
                
                if pred_df[f'{stock_id}'][date] >= threshold:
                    signal[f'{stock_id}'][idx-1] = 1.
            
            idx = idx + 1
    
    get_signal()
    
    get_buy_signals = pd.DataFrame(data=signal, index=datetime_list)
    get_buy_signals = get_buy_signals[start_date:end_date]
    get_buy_signals.index.name = 'date'
    
    return get_buy_signals