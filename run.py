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

def load_config(*paths):
    cfg_dict = {}
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} does not exist.")
        with open(path, "r") as f:
            cfg_dict.update(yaml.safe_load(f))
    return SimpleNamespace(**cfg_dict)

# ←――――――――――――――――――――――――――――――――――――
# 1) 接 argparse，接收 run_all.py 傳進來的 --config
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True,
                    help="Path to the YAML config file")
parser.add_argument('--preprocessor_config', '-p', type=str, default=None,
                    help="Path to the preprocessor YAML config file")
cli_args = parser.parse_args()

# 2) 用參數取代硬編碼
args = load_config(cli_args.config)
print(f"[INFO] Loaded config from {cli_args.config}")
print(f"[INFO] Running with args: {args}")
# ―――――――――――――――――――――――――――――――――――――→

# 3) 如果有 preprocessor_config，則載入預處理器的配置
preprocessor_model = None
preprocessor_args = load_config(cli_args.config, cli_args.preprocessor_config) if cli_args.preprocessor_config else None
if preprocessor_args:
    print(f"[INFO] Loaded preprocessor config from {cli_args.preprocessor_config}")
    print(f"[INFO] Running with preprocessor args: {preprocessor_args}")

    from preprocessors.factory import load_preprocessor_model
    preprocessor_model = load_preprocessor_model(preprocessor_args)

    from data.data_factory import data_provider
    preprocessor_train_set, preprocessor_train_loader = data_provider(preprocessor_args, 'train', isS3E=True)
    preprocessor_val_set, preprocessor_val_loader = data_provider(preprocessor_args, 'val', isS3E=True)
    preprocessor_test_set, preprocessor_test_loader = data_provider(preprocessor_args, 'test', isS3E=True)
    preprocessor_train_val_set, preprocessor_train_val_loader = data_provider(preprocessor_args, 'train_val', isS3E=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("No GPU available")

    from trainer.preprocessor import PreprocessorTrainer
    preprocessor_trainer = PreprocessorTrainer(
        model=preprocessor_model,
        args=preprocessor_args,
        train_set=preprocessor_train_set,
        val_set=preprocessor_train_set,
        test_set=preprocessor_test_set,
        train_val_set=preprocessor_train_val_set,
        train_loader=preprocessor_train_loader,
        val_loader=preprocessor_val_loader,
        test_loader=preprocessor_test_loader,
        train_val_loader=preprocessor_train_val_loader,
        device=device
    )

    best_epoch = preprocessor_trainer.train_with_early_stop()
    preprocessor_trainer.train_final(best_epoch)
    preprocessor_trainer.save_args(f"results/{preprocessor_args.result_file_name}")


from data.data_factory import data_provider

train_set, train_loader = data_provider(args, 'train')
val_set, val_loader = data_provider(args, 'val')
test_set, test_loader = data_provider(args, 'test')
train_val_set, train_val_loader = data_provider(args, 'train_val')

# for debugging purposes, print the first item of each dataset
test_set[0]

from models.factory import load_main_model

args.num_stocks = test_set.get_num_stocks()
model = load_main_model(args, preprocessor_model=preprocessor_model)

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