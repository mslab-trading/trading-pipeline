# pip install streamlit
# streamlit run run_streamlit.py --server.port 8080

import streamlit as st
import pandas as pd
import datetime
from evaluation.stats import sharpe, sharpe_0050

def get_returns(category, strategy, start_date: datetime, end_date: datetime):
    returns = pd.read_csv(f"results_backtest_v2/{category}_{strategy}/returns.csv")
    returns.set_index("date", inplace=True)
    returns.index = pd.to_datetime(returns.index)
    returns = returns[returns.index >= start_date]
    returns = returns[returns.index <= end_date]
    returns = returns / returns.iloc[0]
    return returns

def get_metrics(returns):
    rois = returns.iloc[-1] - 1
    sharpe_model = sharpe(returns["model"])
    sharpe_2330_based = sharpe_0050(returns["model"], returns["2330"])
    sharpe_0050_based = sharpe_0050(returns["model"], returns["0050"])

    return {
        "Model ROI": f"{rois['model']:.2%}",
        "2330 ROI": f"{rois['2330']:.2%}",
        "0050 ROI": f"{rois['0050']:.2%}",
        "Model Sharpe": f"{sharpe_model:.2f}",
        "Model Sharpe (2330-based)": f"{sharpe_2330_based:.2f}",
        "Model Sharpe (0050-based)": f"{sharpe_0050_based:.2f}",
    }

def get_portfolio_value(category, strategy, start_date: datetime, end_date: datetime):
    portfolio_value = pd.read_csv(f"results_backtest_v2/{category}_{strategy}/portfolio_value.csv")
    portfolio_value.fillna(0, inplace=True)
    portfolio_value.set_index("date", inplace=True)
    portfolio_value.index = pd.to_datetime(portfolio_value.index)
    portfolio_value = portfolio_value[portfolio_value.index >= start_date]
    portfolio_value = portfolio_value[portfolio_value.index <= end_date]
    portfolio_value = portfolio_value / portfolio_value.iloc[0].sum()
    return portfolio_value

def get_simplified_portfolio_value(portfolio_value, top_n: int):
    top_stocks = (
        portfolio_value.drop(columns=["cash"])
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    simplified_portfolio_value = portfolio_value[["cash"]]
    simplified_portfolio_value[top_stocks] = portfolio_value[top_stocks]
    simplified_portfolio_value["others"] = portfolio_value.drop(
        columns=top_stocks.tolist() + ["cash"]
    ).sum(axis=1)

    return simplified_portfolio_value

def get_signals(category, strategy):
    buy_signals_path = f"results_backtest_v2/{category}_{strategy}/buy_signals.csv"
    try:
        buy_signals = pd.read_csv(buy_signals_path)
        buy_signals.set_index("date", inplace=True)
        buy_signals.index = pd.to_datetime(buy_signals.index)
        buy_signals.sort_index(ascending=False, inplace=True)
    except FileNotFoundError:
        buy_signals = None

    sell_signals_path = f"results_backtest_v2/{category}_{strategy}/sell_signals.csv"
    try:
        sell_signals = pd.read_csv(sell_signals_path)
        sell_signals.set_index("date", inplace=True)
        sell_signals.index = pd.to_datetime(sell_signals.index)
        sell_signals.sort_index(ascending=False, inplace=True)
    except FileNotFoundError:
        sell_signals = None

    return {"buy_signals": buy_signals, "sell_signals": sell_signals}

def output_planned_trades(tab, strategy: str, buy_signals: pd.DataFrame, sell_signals: pd.DataFrame | None):
    output_len = 5
    if strategy == "allen":
        # Allen
        tab.text("購買股票：model prediction > PR75 且 ADX > 40")
        tab.text("賣出股票：model prediction < PR25")
        for i in range(output_len):
            with tab.container(border=True):
                st.text(buy_signals.index[i].strftime('%Y-%m-%d:'))
                st.text(f"Buy: {buy_signals.columns[buy_signals.iloc[i]].tolist()}")
                st.text(f"Sell: {sell_signals.columns[sell_signals.iloc[i] == True].tolist()}")
                st.text("")
        
    elif strategy == "gino":
        # Gino:
        tab.text("購買股票：model prediction Top1")
        tab.text("賣出股票：放滿 30 天的股票")
        sell_signals = buy_signals.shift(-30, fill_value=False)
        for i in range(output_len):
            with tab.container(border=True):
                st.text(buy_signals.index[i].strftime('%Y-%m-%d:'))
                st.text(f"Buy: {buy_signals.columns[buy_signals.iloc[i] == True].tolist()}")
                st.text(f"Sell: {sell_signals.columns[sell_signals.iloc[i] == True].tolist()}")

    elif strategy == "daily":
        # Daily:
        tab.text("提供交易日收盤時的股票配置比例建議")
        buy_signals = buy_signals[buy_signals >= 0.02]
        buy_signals["cash"] = 1.0 - buy_signals.drop(columns=["cash"]).sum(axis=1)
        for i in range(output_len):
            with tab.container(border=True):
                st.text(buy_signals.index[i].strftime('%Y-%m-%d:'))
                df = buy_signals.loc[buy_signals.index[i]:buy_signals.index[i]]
                st.bar_chart(df, horizontal=True, height=150, stack=True)
                st.write(df)

                                   

# main
st.write("Backtest Dashboard (Streamlit)")

all_categories = ["Top50", "Top100", "Top50_RAM"]
all_strategies = ["allen", "gino", "daily"]
date_range = [datetime.date(2021, 1, 1), datetime.date.today()]
default_date_range = (datetime.date(2025, 6, 1), date_range[1])

with st.container(border=True):
    category = st.selectbox("Category", all_categories)
    strategy = st.selectbox("Strategy", all_strategies)
    # top_n = st.slider("Top N Stocks", min_value=1, max_value=20, value=5)
    start_date, end_date = st.slider("Date", *date_range, default_date_range)
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)


tab1, tab2, tab3 = st.tabs(["Returns", "Portfolio Chart", "Buy & Sell Signals"])

# Tab 1
returns = get_returns(category, strategy, start_date, end_date)
tab1.line_chart(returns, height=250)
metrics = get_metrics(returns)
tab1.text(f"Metrics from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:")
tab1.table(metrics)

# Tab 2
portfolio_value = get_portfolio_value(category, strategy, start_date, end_date)
tab2.area_chart(portfolio_value, stack=True, width="stretch")

with tab2.container(border=True):
    top_n = st.slider("Top N Stocks", min_value=1, max_value=20, value=5)
    simplified_portfolio_value = get_simplified_portfolio_value(portfolio_value, top_n)

tab2.area_chart(simplified_portfolio_value, stack=True)

# Tab 3
signals = get_signals(category, strategy)

tab3.subheader("Planned Trades:")

output_planned_trades(tab3, strategy, signals["buy_signals"], signals["sell_signals"])

tab3.subheader("Raw Buy & Sell Signals")
if signals["buy_signals"] is not None:
    tab3.text("Buy Signals")
    tab3.dataframe(signals["buy_signals"], height=250, width='stretch')
if signals["sell_signals"] is not None:
    tab3.text("Sell Signals")
    tab3.dataframe(signals["sell_signals"], height=250, width='stretch')
