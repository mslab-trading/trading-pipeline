from finlab import data
from pandas import MultiIndex
import pandas as pd
from strategies.backtest.allen import *

#  Input: Target = [stock ids]
# Output: Target = [stock ids] but filtered out those we do not have their data.
def filter_bad_targets(Target, cfg):
    bad = []
    for target in Target:
        try:
            df_stock = pd.read_csv(f"{cfg['root_path']}/{target}.csv")
        except:
            print(f"{target} is bad")
            bad.append(target)
            continue
        df_stock["high_grow"] = df_stock['etl:adj_high'] / df_stock['etl:adj_close'].shift(1)
        df_stock[ "low_grow"] =  df_stock['etl:adj_low'] / df_stock['etl:adj_close'].shift(1)
    for b in bad:
        Target.remove(b)
    return Target

#  Input: Given Target = [stock ids]
# Output: df with the format for backtest
def get_price_df(Target):
    open = data.get('etl:adj_open')[Target]
    close = data.get('etl:adj_close')[Target]
    
    close = pd.DataFrame(close)
    open = pd.DataFrame(open)
    
    open.columns = pd.MultiIndex.from_product([open.columns, ['open']], names=['Ticker', 'Price Type'])
    close.columns = pd.MultiIndex.from_product([close.columns, ['close']], names=['Ticker', 'Price Type'])
    
    price_df = pd.concat([close, open], axis=1)
    price_df = price_df.sort_index(axis=1, level=1) 
    price_df = price_df.sort_index(axis=1, level=0, sort_remaining=True) 
    
    price_df.columns = MultiIndex.from_tuples(
        [(col[0], 'Close' if col[1] == 'close' else 'Open') for col in price_df.columns]
    )
    price_df.fillna(method="bfill", inplace=True)
    return price_df