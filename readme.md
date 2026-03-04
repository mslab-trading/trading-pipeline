# readme
This is the source code for AI Trading.


## Environment Setup

This application runs in Python 3.10. You are recommended to use `conda` or `miniconda` to maintain the virtual environment. Basically, the environment can be compatible with `trading-model`.

```bash
conda create -n pipeline10 python=3.10
```

```bash
pip install -r requirements.txt
```

For calculating stock indicators, TA-LIB is required.
```bash
conda install -c conda-forge libta-lib
conda install -c conda-forge ta-lib
```

## File Architecture
* `config`  
  It contains the configurations for the model.
* `data`  
  It contains the data used for the model.
  * `info`  
    The list of brokers or stocks for each categories.
  * `raw`  
    The raw financial data from Finlab. It can be fetched with `data/fetch_data.py`.
    * `broker`
      Broker's trading history.
    * `global`
      Global indicators
    * `market`
      Market indicators (e.g. prices and investor movements)
* `evaluation`  
  It contains metrics for evaluation, like sharpe ratio.
* `models`  
  It contains the model class files.
* `results`  
  It contains the model output.
  * `Example_Result`  
    An output example of the model output.
* `run_all_python`  
  It contains scripts to train/inference the model for experiments
* `strategies`  
  It contains the strategies used in the backtest, like `daily` and `gino`.

## Finlab Key
To download data from finlab, it is required to have a finlab api key, and store the key as an environment variable.

```bash
vim ~/.bashrc
```

```
export FINLAB_API_KEY= "YOUR_API_KEY";
```

## Fetch Data 

1. cd to `data`
2. run `python3 fetch_data.py` to fetch the up-to-date financial data from Finlab.


## Run Model
The scripts in `run_all_python`.

* `run_all_python`  
  This contains scripts for experiments.

  For example, to train the categories `Top50`, `Top100`, and `Selected` during different year, please run:
  ```bash
  python3 run_all_python/run_all.py
  ```

  The result will be stored in the `results` directory (e.g. `Top50_RAM_Lccc_Dataset_Abs`). In `pred_pct.csv`, it contains the model's predicted max_roi for each stock.

## Backtest / Run Evaluation
In `run_backtest.ipynb`, it will show the roi and invest ratio.
In `run_backtest.py`, you can run backtest and output `returns`, `trades`, and ``