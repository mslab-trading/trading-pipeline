#!/usr/bin/env python3

import os

# 4 种 split 设置
import subprocess, os
from time import sleep
import pandas as pd

# download latest data
def download_data():
    current_cwd = os.getcwd()
    os.chdir(os.path.join(current_cwd, "data"))
    print("Downloading latest data...")
    subprocess.run([
        "python3", "fetch_data.py"],
        stdout=open("fetch_data.log", "w"),
        stderr=subprocess.STDOUT
    )
    print("Data downloaded.")
    os.chdir(current_cwd)

# predict the data
def run_model():
    def run_trading_pipeline_model():
        print("Running trading-pipeline model...")
        subprocess.run(["python3", "run_daily_python/run_model.py"])
        print("Model run complete.")

    def run_trading_model_model():
        current_cwd = os.getcwd()
        os.chdir(os.path.join(current_cwd, "../trading_competition"))

        print("Fetching trading-model data...")
        subprocess.run(["bash", "script/1_load_data.sh"])
        print("Trading-model data fetched.")

        print("Running trading-model model...")
        subprocess.run(["bash", "script/4_test_all_model.sh"])
        print("Trading-model model run complete.")

        os.chdir(current_cwd)

    run_trading_pipeline_model()
    run_trading_model_model()

def convert_model_results():
    print("Converting model results...")
    subprocess.run(["python3", "run_daily_python/convert_model_results.py"])
    print("Model results converted.")

def run_backtest():
    print("Running backtest...")
    subprocess.run(["python3", "run_daily_python/run_backtest.py"])
    subprocess.run(["python3", "run_daily_python/run_backtest_streamlit.py"])
    subprocess.run(["python3", "run_daily_python/run_signals_streamlit.py"])
    print("Backtest run complete.")

def main():
    while True:
        try:
            download_data()
            run_model()
            convert_model_results()
            run_backtest()
        except Exception as e:
            print(f"An error occurred: {e}")
        print("Sleep 1 hour...")
        sleep(60 * 60) # run every hour


if __name__ == "__main__":
    main()
