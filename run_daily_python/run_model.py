#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.getcwd()))

# 4 种 split 设置
import subprocess, yaml, tempfile, os, sys
from time import sleep
import pandas as pd
import itertools

from data.data_dates import get_next_trading_date_str

split_sets = [
    ["2023-01-01", "2025-01-01", "2026-01-01", get_next_trading_date_str()]
]

# 5 个 category
categories = ["Top50", "Top50_RAM", "Top100"]

# 载入 base config
with open("config/main_training.yaml") as f:
    base_cfg = yaml.safe_load(f)

with open("config/preprocessor_training.yaml") as f:
    base_preprocessor_cfg = yaml.safe_load(f)


for cat, splits in itertools.product(categories, split_sets):
    success = False
    while not success:
        for gpu_id in range(7):
            try:
                cfg = base_cfg.copy()
                cfg["split_dates"] = splits
                cfg["category"] = cat
                cfg["result_file_name"] = f"{cat}_{cfg['data']}"

                preprocessor_config = cfg.copy()
                preprocessor_config.update(base_preprocessor_cfg)

                # 写入临时文件
                fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
                with os.fdopen(fd, "w") as tmpf:
                    yaml.safe_dump(cfg, tmpf)

                fd, tmp_preprocessor_path = tempfile.mkstemp(suffix=".yaml")
                with os.fdopen(fd, "w") as tmpf:
                    yaml.safe_dump(preprocessor_config, tmpf)

                print(f">>> Running splits={splits}, category={cat}")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                subprocess.run(
                    [
                        "python",
                        "run.py",
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
