#!/usr/bin/env python3
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 4 种 split 设置
import subprocess, yaml, tempfile, os
from time import sleep
import pandas as pd

r_bound = pd.Timestamp.now() + pd.tseries.offsets.BDay(1) + pd.Timedelta(days=1) # next business day & closed-open interval in split_sets
r_bound_str = r_bound.strftime("%Y-%m-%d")

split_sets = [
    ["2018-01-01", "2020-01-01","2021-01-01","2022-01-01"],
    ["2019-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
    ["2020-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
    ["2021-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
    ["2022-01-01", "2024-01-01", "2025-01-01", r_bound_str
     ]
]

# 5 个 category
categories = ["Top50", "Top100", "Selected"]

# [ 'Top50', 'Top100', 'BioMedTSE', 'ChemicalTSE', 'ElectronicComponentTSE', 'FinanceTSE', 'OptoTSE', "SemiconductorTSE",
#                   "Selected", "SelectedVer2", "BuildingTSE", "CarTSE", "ClothesTSE", "NetworkTSE", "MetalTSE", "EETSE", "ComputerTSE"]
# 载入 base config
with open("config/main_training.yaml") as f:
    base_cfg = yaml.safe_load(f)

with open("config/preprocessor_training.yaml") as f:
    base_preprocessor_cfg = yaml.safe_load(f)

for cat in categories:
    for splits in split_sets:
        success = False
        while not success:
            try:
                cfg = base_cfg.copy()
                cfg["split_dates"] = splits
                cfg["category"] = cat
                cfg["result_file_name"] = f"{cat}_{cfg['data']}"

                preprocessor_config = cfg.copy()
                preprocessor_config.update(base_preprocessor_cfg)
                preprocessor_config["result_file_name"] = (
                    f"{cat}_{preprocessor_config['data']}"
                )

                # 写入临时文件
                fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
                with os.fdopen(fd, "w") as tmpf:
                    yaml.safe_dump(cfg, tmpf)

                fd, tmp_preprocessor_path = tempfile.mkstemp(suffix=".yaml")
                with os.fdopen(fd, "w") as tmpf:
                    yaml.safe_dump(preprocessor_config, tmpf)

                print(f">>> Running splits={splits}, category={cat}")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "3"
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
            except:
                sleep(60)
                print("An error occurred, retrying...")
                continue
