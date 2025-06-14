# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# %%
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

args = load_config("config/training_config.yaml")

# %%
from data.data_factory import data_provider

train_set, train_loader = data_provider(args, 'train')
val_set, val_loader = data_provider(args, 'val')
test_set, test_loader = data_provider(args, 'test')
train_val_set, train_val_loader = data_provider(args, 'train_val')

# %%
from models.load_model import load_model

args.num_stocks = test_set.get_num_stocks()
model = load_model(args)

# %%
from trainer.stock_trainer import Trainer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("No GPU available")

trainer = Trainer(
    model=model,
    args=args,
    train_set=train_set,
    val_set=val_set,
    test_set=test_set,
    train_val_set=train_val_set,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    train_val_loader=train_val_loader,
    device=device
)

best_epoch = trainer.train_with_early_stop()
trainer.train_final(best_epoch)

# %%
import pandas as pd
test_year = f"{pd.to_datetime(args.split_dates[2]).year}"
checkpoint_path = f"checkpoints/{args.model_name}_{args.category}_{test_year}_{args.loss}.pt"
trainer.save_checkpoint(checkpoint_path)

# %%
trainer.evaluate_and_save(f"results/{args.result_file_name}")
# %%
# Simple analysis
import pandas as pd
import numpy.ma as ma

pred_df = pd.read_csv(f"results/{args.result_file_name}/{test_year}/test/pred_pct.csv")
truth_df = pd.read_csv(f"results/{args.result_file_name}/{test_year}/test/truth_pct.csv")

pred_df.set_index("date", inplace=True)
truth_df.set_index("date", inplace=True)

pred_values = pred_df.values.flatten()
truth_values = truth_df.values.flatten()

corr = ma.corrcoef(ma.masked_invalid(pred_values), ma.masked_invalid(truth_values))[0][1]
print(f"Correlation between predictions and truth: {corr:.4f}")

# %%
import sys
import importlib

# Add the parent directory to sys.path (adjust as needed)
sys.path.append("..")  # or "." if utils is in the same directory as your notebook

# Now import
import utils.io
importlib.reload(utils.io)

from utils.io import read_all_df

test_pred, test_truth, val_pred, val_truth = read_all_df(f"results/{args.category}", start_year=2021, end_year=2024)
test_pred

# %%
backtest_args = load_config("config/backtest_config.yaml")
backtest_args


