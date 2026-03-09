import os
import subprocess

for model in ["StockAttentioner", "BasicModel", "iTransformer"]:
    for category in ["Top50", "Top50_RAM", "Top100"]:
        for backtest_type in ["allen", "gino", "gino_open", "daily"]:
            print(f">>> Backtesting model={model}, category={category}, backtest_type={backtest_type}")
            f = open("config/backtest.yaml")
            env = os.environ.copy()
            # env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            subprocess.run(
                [
                    "python",
                    "run_backtest.py",
                    "--config",
                    "config/backtest.yaml",
                    "--backtest_type",
                    backtest_type,
                    "--start",
                    "2021-01-01",
                    "--input_dir",
                    f"results/{model}_{category}_Dataset_Abs",
                    "--output_dir",
                    f"results_backtest_streamlit/{model}/{category}/{backtest_type}"
                ],
                check=True,
                env=env,
                stdout=open("run_backtest.log", "w"),
                stderr=subprocess.STDOUT,
            )
