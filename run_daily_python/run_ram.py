#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.getcwd()))

from data.data_dates import get_next_trading_date


# 4 种 split 设置
import subprocess, yaml, tempfile, os, sys
from time import sleep
import pandas as pd
import itertools

next_trading_date = get_next_trading_date().strftime("%Y-%m-%d")
split_sets = [
    # ["2018-01-01", "2020-01-01", "2021-01-01", "2022-01-01"],
    # ["2019-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
    # ["2020-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
    # ["2021-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
    ["2022-01-01", "2024-01-01", "2025-01-01", "2026-01-01"],
    # ["2023-01-01", "2025-01-01", "2026-01-01", next_trading_date]
]

# 5 个 category
categories = ["Top50_RAM"]

# 载入 config
with open("config/preprocessor_training.yaml") as f:
    base_preprocessor_cfg = yaml.safe_load(f)

for model in ["StockAttentioner", "BasicModel"]:
    if model == "StockAttentioner":
        base_cfg_path = "config/main_stock_attentioner.yaml"
    elif model == "BasicModel":
        base_cfg_path = "config/main_basic_model.yaml"
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    for cat, splits in itertools.product(categories, split_sets):
        success = False
        while not success:
            for gpu_id in range(7):
                try:
                    cfg = base_cfg.copy()
                    cfg["split_dates"] = splits
                    cfg["category"] = cat
                    cfg["result_file_name"] = f"{model}_{cat}_{cfg['data']}"

                    preprocessor_config = cfg.copy()
                    preprocessor_config.update(base_preprocessor_cfg)

                    # 写入临时文件
                    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
                    with os.fdopen(fd, "w") as tmpf:
                        yaml.safe_dump(cfg, tmpf)

                    fd, tmp_preprocessor_path = tempfile.mkstemp(suffix=".yaml")
                    with os.fdopen(fd, "w") as tmpf:
                        yaml.safe_dump(preprocessor_config, tmpf)

                    # Remove old results
                    subprocess.run(["rm", "-rf", f"results/{cfg['result_file_name']}/20260101_*"])

                    print(f">>> Running splits={splits}, category={cat}")

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    subprocess.run(
                        [
                            "python",
                            "run.py",
                            # "--load_checkpoint",
                            "--config",
                            tmp_path,
                            # 如果需要 preprocessor_config，可以取消注釋
                            # "--preprocessor_config", tmp_preprocessor_path
                        ],
                        check=True,
                        env=env,
                    )

                    os.remove(tmp_path)
                    success = True
                    print("Successfully completed for splits:", splits, "category:", cat)
                    break
                except Exception as e:
                    print(f"An error occurred in gpu({gpu_id}), retrying different gpu... Error: {e}")
                    sleep(10)
                    continue
            if not success:
                print("All gpus are busy, waiting for 1 minute...")
                sleep(60)
