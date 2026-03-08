import os
import subprocess

for category in ["Top50", "Top50_RAM", "Top100"]:
    for backtest_type in ["allen", "gino", "daily"]:
        print(f">>> Running category={category}, backtest_type={backtest_type}")
        f = open("config/backtest.yaml")
        env = os.environ.copy()
        # env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run(
            [
                "python",
                "run_signals.py",
                "--config",
                "config/backtest.yaml",
                "--backtest_type",
                backtest_type,
                "--start",
                "2021-01-01",
                "--input_dir",
                f"results/{category}_Dataset_Abs",
                "--output_dir",
                f"results_backtest_v2/{category}_{backtest_type}"
            ],
            check=True,
            env=env,
            stdout=open("run_signals.log", "w"),
            stderr=subprocess.STDOUT,
        )
