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
    print("Running model...")
    subprocess.run(["python3", "run_daily_python/run_model.py"])
    print("Model run complete.")

def run_signals():
    print("Running backtest...")
    subprocess.run(["python3", "run_daily_python/run_signals.py"])
    print("Backtest run complete.")

def run_backtest():
    print("Running backtest...")
    subprocess.run(["python3", "run_daily_python/run_backtest.py"])
    subprocess.run(["python3", "run_daily_python/run_backtest_v2.py"])
    print("Backtest run complete.")

def copy_backtest_results():
    print("Copying backtest results...")
    subprocess.run(["cp", "-r", "results_backtest/", "web/"])
    print("Backtest results copied.")


def main():
    download_data()
    run_model()
    run_backtest()
    copy_backtest_results()

if __name__ == "__main__":
    main()
