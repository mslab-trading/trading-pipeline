"""
Prerequisites for talib:
https://pypi.org/project/TA-Lib/
conda install -c conda-forge libta-lib
conda install -c conda-forge ta-lib
pip install TA-Lib
"""


import finlab
from finlab import data
import pandas as pd
import os
from tqdm import tqdm

finlab.login("ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m")

BROKER_PATH = "raw/broker/"
GLOBAL_PATH = "raw/global/"
MARKET_PATH = "raw/market/"

universe_market = "TSE"

market_features = [
    "etl:adj_close",
    "etl:adj_open",
    "etl:adj_high",
    "etl:adj_low",
    "price:成交筆數",
    "price:成交金額",
    "foreign_investors_shareholding:全體外資及陸資持股比率",
    "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
    "institutional_investors_trading_summary:外資自營商買賣超股數",
    "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)",
    "institutional_investors_trading_summary:自營商買賣超股數(避險)",
]
technical_indicators = [
    "RSI",
    "SMA",
    "MACD",
    "CMO",
    "ATR",
    "BBANDS_upper",
    "BBANDS_mid",
    "BBANDS_lower",
]


def get_market_dfs():
    print("Fetching market features...")
    market_dfs = {}
    with data.universe(universe_market):
        for market_feature in tqdm(market_features):
            market_dfs[market_feature] = data.get(market_feature)

    return market_dfs


def get_technical_dfs():
    print("Calculating technical features...")
    technical_dfs = {}
    with data.universe(universe_market):
        for indicator in tqdm(technical_indicators):
            if indicator.startswith("BBANDS"):
                upper, middle, lower = data.indicator("BBANDS")
                technical_dfs["BBANDS_upper"] = upper
                technical_dfs["BBANDS_mid"] = middle
                technical_dfs["BBANDS_lower"] = lower
            elif indicator.startswith("MACD"):
                macd, signal, hist = data.indicator("MACD")
                technical_dfs["MACD"] = hist
            else:
                technical_dfs[indicator] = data.indicator(
                    indicator
                )  # Precompute indicators

    return technical_dfs


def fetch_broker_data():
    """
    Fetch the latest broker data.
    """
    print("Fetching broker transactions data...")
    with data.universe(universe_market):
        broker_transactions = data.get("broker_transactions")  # takes about 5 minutes
    stock_list = broker_transactions["stock_id"].unique().tolist()

    os.makedirs(BROKER_PATH, exist_ok=True)
    print("Saving broker data...")
    for stock in tqdm(stock_list):
        sub_df = broker_transactions[broker_transactions["stock_id"] == stock]
        sub_df = sub_df[["date", "broker", "buy", "sell"]]
        sub_df = sub_df.sort_values(by="date")
        sub_df.to_csv(f"{BROKER_PATH}/{stock}.csv", index=False)


def fetch_global_data():
    """
    Fetch the latest global market data from market and indicators.
    """
    world_index_features = ["^VIX", "^IXIC", "^GSPC", "^TWII"]
    world_df = data.get("world_index:adj_close")
    world_df = world_df[world_df.index >= "2018-01-01"]
    world_df = world_df[world_index_features]
    world_df = world_df.dropna()

    os.makedirs(GLOBAL_PATH, exist_ok=True)
    world_df.to_csv(f"{GLOBAL_PATH}/global_data.csv")


def fetch_market_data():
    """
    Fetch the latest market data.
    """

    # fetch stock list
    with data.universe(universe_market):
        close_df = data.get("etl:adj_close")
        close_df = close_df[close_df.index >= "2018-01-01"]
        stock_list = close_df.columns.tolist()

    market_dfs = get_market_dfs()
    technical_dfs = get_technical_dfs()

    dates = close_df.index
    os.makedirs(MARKET_PATH, exist_ok=True)

    print("Saving market data...")
    for stock in tqdm(stock_list):
        stock_df = pd.DataFrame(index=dates, columns=market_features + technical_indicators)
        for market_feature in market_features:
            if stock not in market_dfs[market_feature].columns:
                break
            stock_df[market_feature] = market_dfs[market_feature][stock]
        for technical_indicator in technical_indicators:
            if stock not in technical_dfs[technical_indicator].columns:
                break
            stock_df[technical_indicator] = technical_dfs[technical_indicator][stock]
        stock_df = stock_df.dropna()
        if len(stock_df) == 0:
            print(f"[Warning] {stock} has no data after dropping NA. Skipping...")
            continue
        stock_df.to_csv(f"{MARKET_PATH}/{stock}.csv")

def main():
    fetch_broker_data()
    fetch_global_data()
    fetch_market_data()


if __name__ == "__main__":
    main()
