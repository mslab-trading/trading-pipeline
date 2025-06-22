#!/usr/bin/env python3
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import yaml
from types import SimpleNamespace
import argparse

def load_config(path):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

# ←――――――――――――――――――――――――――――――――――――
# 1) 接 argparse，接收 run_all.py 傳進來的 --config
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True,
                    help="Path to the YAML config file")
cli_args = parser.parse_args()

# 2) 用參數取代硬編碼
args = load_config(cli_args.config)
print(f"[INFO] Loaded config from {cli_args.config}")
print(f"[INFO] Running with args: {args}")
# ―――――――――――――――――――――――――――――――――――――→

from data.data_factory import data_provider

train_set, train_loader = data_provider(args, 'train')
val_set, val_loader = data_provider(args, 'val')
test_set, test_loader = data_provider(args, 'test')
train_val_set, train_val_loader = data_provider(args, 'train_val')

# for debugging purposes, print the first item of each dataset
test_set[0]

from models.factory import load_main_model

args.num_stocks = test_set.get_num_stocks()
model = load_main_model(args, preprocessor_model=None)

from trainer.multi_stock import MultiStockTrainer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise ValueError("No GPU available")

trainer = MultiStockTrainer(
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
trainer.evaluate_and_save(f"results/{args.result_file_name}")