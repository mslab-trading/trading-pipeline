"""
This script is designed to be run as a daily cron job. It will:
1. Check if today is a trading day. If not, it will not run any further steps.
2. Fetch the latest data and run the model after 1 a.m. 
3. Run the backtest and output results after 5 p.m.
4. If anything goes wrong, it will sleep for 30 minutes and try again.
"""

import os
import subprocess
import sys
from time import sleep

import pandas as pd

sys.path.append(os.path.join(os.getcwd()))
from data.data_dates import (
    get_last_trading_date_str,
    get_next_trading_date_str,
    get_today_date_str,
    is_trading_day,
)

# fetch data + run model: 1 a.m.
# backtest + output: 5 p.m.
# sleep: 30 min


# download latest data
def download_data():
    current_cwd = os.getcwd()
    os.chdir(os.path.join(current_cwd, "data"))
    print_log("Downloading latest data...")
    subprocess.run(
        ["python3", "fetch_data.py"],
        stdout=open("cron_data_fetch.log", "w"),
        stderr=subprocess.STDOUT,
    )
    print_log("Data downloaded.")
    os.chdir(current_cwd)


def check_data():
    market_data_path = "data/raw/market/2330.csv"
    broker_data_path = "data/raw/broker/2330.csv"

    if not os.path.exists(market_data_path):
        print_log(f"Market data file {market_data_path} does not exist.")
        return False
    market_df = pd.read_csv(market_data_path)
    if get_last_trading_date_str() not in market_df["date"].values:
        print_log(
            f"Last trading date {get_last_trading_date_str()} not in market data."
        )
        return False

    if not os.path.exists(broker_data_path):
        print_log(f"Broker data file {broker_data_path} does not exist.")
        return False
    broker_df = pd.read_csv(broker_data_path)
    if get_last_trading_date_str() not in broker_df["date"].values:
        print_log(
            f"Last trading date {get_last_trading_date_str()} not in broker data."
        )
        return False

    print_log("Data check complete.")
    return True


def check_model_results():
    next_trading_date = get_next_trading_date_str().replace("-", "")
    model_results_path = (
        f"results/Top100_Dataset_Abs/20260101_{next_trading_date}/test/pred_pct.csv"
    )
    if not os.path.exists(model_results_path):
        print_log(f"Model results file {model_results_path} does not exist.")
        return False
    print_log(f"Model results file {model_results_path} exists.")
    return True


def check_backtest_results():
    today = get_today_date_str()
    returns_path = f"results_backtest/Top100_daily/returns.csv"
    if not os.path.exists(returns_path):
        print_log(f"Backtest results file {returns_path} does not exist.")
        return False
    returns = pd.read_csv(returns_path)
    if today not in returns["date"].values:
        print_log(f"Today {today} not in backtest results.")
        return False
    if not os.path.exists(returns_path):
        print_log(f"Backtest results file {returns_path} does not exist.")
        return False
    return True


def run_model():
    print_log("Running model...")
    subprocess.run(["python3", "run_daily_python/run_model.py"])
    print_log("Model run complete.")


def run_backtest():
    print_log("Running backtest...")
    subprocess.run(["python3", "run_daily_python/run_backtest.py"])
    print_log("Backtest run complete.")

def copy_backtest_results():
    print_log("Copying backtest results...")
    subprocess.run(["cp", "-r", "results_backtest/", "web/"])
    print_log("Backtest results copied.")


def get_timestamp_str():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def print_log(message):
    print(f"[{get_timestamp_str()}] {message}")


def main():
    while True:
        if is_trading_day(pd.Timestamp.now()):
            if pd.Timestamp.now().hour >= 1:
                # if is after 1 a.m.
                while not check_data():
                    # if data file is not up-to-date:
                    download_data()
                    if not check_data():
                        print(
                            f"[{get_timestamp_str()}] Data not ready. Sleeping for 30 minutes..."
                        )
                        sleep(30 * 60)  # 30 minutes
                        continue
                print_log("Data is ready.")

                while not check_model_results():
                    # while model results are not up-to-date:
                    run_model()
                    if not check_model_results():
                        print(
                            f"[{get_timestamp_str()}] Model results not ready. Sleeping for 30 minutes..."
                        )
                        sleep(30 * 60)  # 30 minutes
                print_log("Model results are ready.")
            else:
                print_log("Not time to run model yet (1 a.m.).")

            if pd.Timestamp.now().hour >= 17:
                # if is after 5 p.m.
                while not check_backtest_results():
                    # while today's backtest results do not exist:
                    run_backtest()
                    if not check_backtest_results():
                        print_log(
                            "Backtest results not ready. Sleeping for 30 minutes..."
                        )
                        sleep(30 * 60)  # 30 minutes
                    else:
                        copy_backtest_results()
                print_log("Backtest results are ready.")
            else:
                print_log("Not time to run backtest yet (5 p.m.).")
        else:
            print_log("Not a trading day.")
        print_log("Sleeping for 1 hour...")
        sleep(60 * 60)  # check every hour


if __name__ == "__main__":
    main()
