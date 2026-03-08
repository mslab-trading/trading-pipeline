# This file defines the trading calendar for the year 2026.
# Please update the holidays list and the start_date/end_date if you want to use it for other years.
import pandas as pd
import datetime
import os
import finlab
from finlab import data

finlab_token = os.getenv("FINLAB_API_KEY")
finlab.login(finlab_token)

# source: https://www.twse.com.tw/holidaySchedule/holidaySchedule?response=html
holidays = [
    datetime.datetime(2026, 1, 1),
    datetime.datetime(2026, 2, 12),
    datetime.datetime(2026, 2, 13),
    datetime.datetime(2026, 2, 16),
    datetime.datetime(2026, 2, 17),
    datetime.datetime(2026, 2, 18),
    datetime.datetime(2026, 2, 19),
    datetime.datetime(2026, 2, 20),
    datetime.datetime(2026, 2, 27),
    datetime.datetime(2026, 4, 3),
    datetime.datetime(2026, 4, 6),
    datetime.datetime(2026, 5, 1),
    datetime.datetime(2026, 6, 19),
    datetime.datetime(2026, 9, 25),
    datetime.datetime(2026, 9, 28),
    datetime.datetime(2026, 10, 9),
    datetime.datetime(2026, 10, 10),
    datetime.datetime(2026, 10, 26),
    datetime.datetime(2026, 12, 25),
]

trading_cbday = pd.offsets.CustomBusinessDay(
    holidays=holidays,
    weekmask="Mon Tue Wed Thu Fri",
)
start_date = pd.Timestamp("2026-01-01")
end_date = pd.Timestamp("2026-12-31")
trading_dates_2026 = pd.bdate_range(start_date, end_date, freq=trading_cbday)

def get_all_trading_dates():
    trading_dates_historical = data.get("etl:adj_close").index
    trading_dates = set(trading_dates_historical) | set(trading_dates_2026)
    trading_dates = sorted(list(trading_dates))
    trading_dates = pd.to_datetime(trading_dates)
    return trading_dates

trading_dates = get_all_trading_dates()

def is_trading_date(date: pd.Timestamp) -> bool:
    date = date.floor("D") # only compare date, ignore time
    return date in trading_dates.values

def get_next_trading_date(date: pd.Timestamp = None) -> pd.Timestamp:
    if date is None:
        date = pd.Timestamp.now().floor("D")
    next_trading_days = trading_dates[trading_dates > date]
    if next_trading_days.empty:
        return None
    return next_trading_days[0]

def get_last_trading_date(date: pd.Timestamp = None) -> pd.Timestamp:
    if date is None:
        date = pd.Timestamp.now().floor("D")
    past_trading_days = trading_dates[trading_dates < date]
    if past_trading_days.empty:
        return None
    return past_trading_days[-1]