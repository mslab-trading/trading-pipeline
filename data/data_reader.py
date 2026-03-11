from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


def read_market_data(
    root_dir: Union[str, Path],
    stock_ids: List[str],
    global_data_path: Optional[Union[str, Path]] = None,
    market_features: Optional[List[str]] = None,
    global_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load market data for given stock_ids from CSVs in root_dir.
    Optionally merge global indicators by date.
    Returns a DataFrame with 'date' as datetime and 'stock_id' as an additional column.
    """
    market_df = _load_and_combine_csvs(root_dir, stock_ids)
    
    if market_features:
        market_features += ['date', 'stock_id']  # Ensure 'date' and 'stock_id' are included
        market_features = list(dict.fromkeys(market_features)) # Remove duplicates
        missing_features = [feat for feat in market_features if feat not in market_df.columns]
        if missing_features:
            raise ValueError(f"The following market features are missing: {missing_features}")
        market_df = market_df[market_features]
    
    market_df['date'] = pd.to_datetime(market_df['date'])

    if global_data_path:
        global_df = read_global_data(global_data_path, global_features)
        market_df = (market_df.join(global_df, how='left', on='date'))
    
    return market_df


def read_broker_data(
    root_dir: Union[str, Path],
    stock_ids: List[str],
    broker_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load broker data for given stock_ids and compute net buy-sell per broker.
    Returns a DataFrame with 'date' and one column per broker.
    """
    return _load_and_combine_csvs(root_dir, stock_ids, broker=True, broker_names=broker_names)


def read_global_data(
    global_data_path: Union[str, Path],
    global_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load global market indicators from CSV and index by date.
    Assumes CSV contains a 'date' column.
    """
    path = Path(global_data_path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    
    if global_features:
        missing_features = [feat for feat in global_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from global data: {missing_features}")
        df = df[global_features]
    
    return df


def _load_and_combine_csvs(
    root_dir: Union[str, Path],
    stock_ids: List[str],
    broker: bool = False,
    broker_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Helper to load individual CSVs for each stock_id and concatenate.
    If broker=True, applies broker preprocessing.
    """
    dfs = []
    root = Path(root_dir)
    
    for sid in stock_ids:
        file_path = root / f"{sid}.csv"
        if not file_path.exists():
            print(f"[Warning] {file_path.name} not found in {root}.")
            continue

        df = pd.read_csv(file_path, parse_dates=["date"])
        if broker:
            df = _broker_preprocessing(df, broker_names)
        df["stock_id"] = sid
        dfs.append(df)

    if not dfs:
        breakpoint()
        raise ValueError(f"No CSVs loaded from {root} for stock_ids {stock_ids}")
    return pd.concat(dfs, ignore_index=True)


def _broker_preprocessing(df: pd.DataFrame, broker_names: List[str]) -> pd.DataFrame:
    """
    Compute net buy-sell for each broker.
    Assumes columns like '<broker>-buy' and '<broker>-sell'.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    df['diff'] = df['buy'] - df['sell']
    df.drop(columns=['buy', 'sell'], inplace=True)
    df = df[df['broker'].isin(broker_names)]

    # breakpoint()
    
    df_wide = df.pivot_table(index='date', columns='broker', values='diff')

    for broker in broker_names:
        if broker not in df_wide.columns:
            df_wide[broker] = 0
    df_wide = df_wide.reset_index()
    return df_wide


if __name__ == "__main__":
    root_path = "./data/raw/market/"
    broker_path = "./data/raw/broker/"
    general_data_path = "./data/raw/global/global_data.csv"
    
    stock_ids =  ['2330', '2317', '2454']  # Example stock IDs
    market_features = ['etl:adj_close','etl:adj_open','etl:adj_high','etl:adj_low','price:成交筆數']
    global_features = ["^VIX", "PCR_Volume"]
    
    broker_names = [
        "凱基-台北",
        "元大證券",
        "摩根大通",
        "新加坡商瑞銀",
        "美林",
        "台灣摩根士丹利",
        "美商高盛",
        "瑞士信貸",
        "永豐金證券",
        "富邦證券",
    ]
    
    market_df = read_market_data(
        root_path,
        stock_ids,
        global_data_path=general_data_path,
        market_features=market_features,
        global_features=global_features
    )
    
    global_df = read_global_data(general_data_path, global_features=global_features)
    broker_df = read_broker_data(broker_path, stock_ids, broker_names)
    
    
