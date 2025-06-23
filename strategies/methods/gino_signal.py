import pandas as pd
import os

def get_buy_signals(pct_dir, store_dir, filename, threshold = 0.0):
    prediction_max = None
    for dir in os.listdir(f'{pct_dir}'):
        if prediction_max is None:
            prediction_max = pd.read_csv(f'{pct_dir}/{dir}/test/pred_pct.csv', index_col=0)
        else:
            tmp = pd.read_csv(f'{pct_dir}/{dir}/test/pred_pct.csv', index_col=0)
            prediction_max = pd.concat([prediction_max, tmp])
    
    prediction_max = prediction_max.sort_index()
    datetime_list = prediction_max.index.tolist()
    stock_names = list(prediction_max.columns)
    
    signal = {}
    for name in stock_names:
        signal[f'{name}'] = [0. for _ in range(len(datetime_list))]
    
    start_date = prediction_max.index[0]
    end_date = prediction_max.index[-1]
    start_idx = datetime_list.index(start_date)
    end_idx = datetime_list.index(end_date)
    
    def get_signal():

        idx = start_idx
        while idx <= end_idx:
            date = datetime_list[idx]
            if date in prediction_max.index:
                stock_id = prediction_max.loc[date].idxmax()
                
                if prediction_max[f'{stock_id}'][date] >= threshold:
                    signal[f'{stock_id}'][idx-1] = 1.
            
            idx = idx + 1
    
    get_signal()
    
    signal = pd.DataFrame(data=signal, index=datetime_list)
    signal = signal[start_date:end_date]
    signal.index.name = 'date'
    signal.to_csv(f'{store_dir}/{filename}')

if __name__ == '__main__':
    store_dir = f'signals'
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
        
    pct_dir = f'results/Example_Result'
    threshold = 0.
    filename = f'signal_{threshold}.csv'
    get_buy_signals(pct_dir, store_dir, filename)