#!/usr/bin/env python3
import os
import subprocess
import yaml
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import copy

# 全局指定給子進程的 GPU（也可以自行改為輪轉分配）
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
MAX_WORKERS = 1

# 4 種 split 設定
split_sets = [
    ["2018-01-01","2020-01-01","2021-01-01","2022-01-01"],
    ["2019-01-01","2021-01-01","2022-01-01","2023-01-01"],
    ["2020-01-01","2022-01-01","2023-01-01","2024-01-01"],
    ["2021-01-01","2023-01-01","2024-01-01","2025-01-01"],
]

# 多個 category
categories = [
    "Selected","SelectedVer2", 'Top50','Top100','ChemicalTSE',
    'FinanceTSE','OptoTSE',"CarTSE","NetworkTSE", "EETSE","ComputerTSE"
]

# 載入 base config once
with open("config/main_training.yaml") as f:
    base_cfg = yaml.safe_load(f)

def run_split(splits, category, loss, broker, concat_market_global, base_cfg):
    """
    為單個 (category, splits) 生成臨時 config 並呼叫 run.py。
    如失敗則 sleep 60 秒後重試，直到成功為止。
    """
    # 準備 config file
    cfg = copy.deepcopy(base_cfg)
    cfg["split_dates"]      = splits
    cfg["category"]         = category
    cfg['loss']            = loss
    cfg["broker"]          = broker
    cfg["concat_market_global"] = concat_market_global
    cfg["result_file_name"] = f"{category}_D{cfg['data']}_L{cfg['loss']}_B{cfg['broker']}_G{cfg['concat_market_global']}"
    
    if concat_market_global:
        cfg["feature_dim"] = 14
    else:
        cfg["feature_dim"] = 9

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as tmpf:
            yaml.safe_dump(cfg, tmpf)

        # 每個子進程都用同一份環境（包含上面設定的 CUDA_VISIBLE_DEVICES）
        env = os.environ.copy()

        # 不斷重試直到成功
        while True:
            try:
                subprocess.run(
                    ["python", "run.py", "--config", tmp_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )
                print(f"[ OK ] {category} | splits={splits}")
                break
            except subprocess.CalledProcessError as e:
                print(f"[ERR] {category} | splits={splits} | exit {e.returncode}")
                print(e.stderr.decode().strip())
                print("→ 60s 後重試…")
                time.sleep(60)
            except Exception as e:
                print(f"[ERR] {category} | splits={splits} | exception: {e}")
                print("→ 60s 後重試…")
                time.sleep(60)

    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    # 全局一次最多 4 個子進程
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for category, loss, broker, concat_market_global in product(
            categories,
            ["mse", "ccc"],
            [0, 1],
            [0, 1]
        ):
            print(f"\n=== Starting category: {category} ===")
            for splits in split_sets:
                futures.append(
                    executor.submit(run_split, splits, category, loss, broker, concat_market_global, base_cfg)
                )
        # 等待所有任務完成（都包含重試邏輯，不會中斷）
        for _ in as_completed(futures):
            pass

    print("\nAll categories done!")
