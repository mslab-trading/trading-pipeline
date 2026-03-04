# pip install streamlit
# streamlit run run_streamlit.py --server.port 8080

import streamlit as st
import pandas as pd

st.write("Backtest Dashboard (Streamlit)")

all_categories = ["Top50", "Top100", "Top50_RAM"]
all_backtests = ["allen", "gino", "daily"]
with st.container(border=True):
    category = st.selectbox("Category", all_categories)
    backtest = st.selectbox("Backtest", all_backtests)
    top_n = st.slider("Top N Stocks", min_value=1, max_value=20, value=5)

# returns chart
returns = pd.read_csv(f"results_backtest/{category}_{backtest}/returns.csv")
returns.set_index("date", inplace=True)
returns.index = pd.to_datetime(returns.index)

# portfolio chart
portfolio_value = pd.read_csv(f"results_backtest/{category}_{backtest}/portfolio_value.csv")
portfolio_value.fillna(0, inplace=True)
portfolio_value.set_index("date", inplace=True)
portfolio_value.index = pd.to_datetime(portfolio_value.index)

# simplified portfolio value
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

tab1, tab2, tab3 = st.tabs(["Returns", "Portfolio Chart", "Dataframe"])
tab1.line_chart(returns, height=250)

tab2.area_chart(portfolio_value, stack=True)
tab2.area_chart(simplified_portfolio_value, stack=True)

tab3.dataframe(returns, height=250, width='stretch')
