#!/usr/bin/env python3
import subprocess
import yaml
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 4 种 split 设置
split_sets = [
    ["2018-01-01","2020-01-01","2021-01-01","2022-01-01"],
    ["2019-01-01","2021-01-01","2022-01-01","2023-01-01"],
    ["2020-01-01","2022-01-01","2023-01-01","2024-01-01"],
    ["2021-01-01","2023-01-01","2024-01-01","2025-01-01"],
]

# 多个 category
categories = [
    'Top50','Top100','BioMedTSE','ChemicalTSE','ElectronicComponentTSE',
    'FinanceTSE','OptoTSE',"SemiconductorTSE","Selected","SelectedVer2",
    "BuildingTSE","CarTSE","ClothesTSE","NetworkTSE","MetalTSE",
    "EETSE","ComputerTSE"
]

# 载入 base config once
with open("config/training_config.yaml") as f:
    base_cfg = yaml.safe_load(f)

def run_split(splits, category, base_cfg):
    """
    为单个 (category, splits) 生成临时 config 并调用 run.py。
    内部捕获所有异常，保证不会中断线程池。
    """
    cfg = base_cfg.copy()
    cfg["split_dates"]      = splits
    cfg["category"]         = category
    cfg["result_file_name"] = f"{category}_{cfg['data']}"

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as tmpf:
            yaml.safe_dump(cfg, tmpf)

        subprocess.run(
            ["python", "run.py", "--config", tmp_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[OK]    {category} | splits={splits}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {category} | splits={splits} | exit {e.returncode}")
        print(e.stderr.decode().strip())
    except Exception as e:
        print(f"[ERROR] {category} | splits={splits} | exception: {e}")
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    for category in categories:
        print(f"\n=== Starting category: {category} ===")
        # 在同一个 category 下并行跑所有 split
        with ThreadPoolExecutor(max_workers=len(split_sets)) as executor:
            futures = [
                executor.submit(run_split, splits, category, base_cfg)
                for splits in split_sets
            ]
            # 等待所有 futures 完成（无论成功或失败）
            for _ in as_completed(futures):
                pass
        print(f"=== Finished category: {category} ===")

    print("\nAll categories done!")
