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


def main():
    download_data()
    run_model()

if __name__ == "__main__":
    main()
