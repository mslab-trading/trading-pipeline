import os
import torch
import numpy as np
import pandas as pd
from utils.early_stop import EarlyStop
import json
from tqdm import tqdm
from trainer.multi_stock import MultiStockTrainer

class PreprocessorTrainer(MultiStockTrainer):
    def __init__(
        self, model, args,
        train_set, val_set, test_set, train_val_set,
        train_loader, val_loader, test_loader, train_val_loader,
        device,
    ):
        super().__init__(
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

    def _prepare_batch(self,
        Xs, Xs_broker, Xs_general,
        X2s, X2s_broker, X2s_general,
        Ys
    ):
        Xs = Xs.to(self.device).float()
        Xs_broker = Xs_broker.to(self.device).float()
        Xs_general = Xs_general.to(self.device).float()
        X2s = X2s.to(self.device).float()   
        X2s_broker = X2s_broker.to(self.device).float()
        X2s_general = X2s_general.to(self.device).float()
        Ys = Ys.to(self.device).float()

        Xs = torch.nan_to_num(Xs, nan=0.0)
        Xs_broker = torch.nan_to_num(Xs_broker, nan=0.0)
        Xs_general = torch.nan_to_num(Xs_general, nan=0.0)
        X2s = torch.nan_to_num(X2s, nan=0.0)
        X2s_broker = torch.nan_to_num(X2s_broker, nan=0.0)
        X2s_general = torch.nan_to_num(X2s_general, nan=0.0)
        Ys = torch.nan_to_num(Ys, nan=0.0)

        return Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys

    def _masked_loss(self, preds, Ys, Xs, Xs_broker, X2s, X2s_broker):
        nan_mask_Xs = torch.isnan(Xs).any(dim=(2, 3))
        nan_mask_Xs_broker = torch.isnan(Xs_broker).any(dim=(2, 3))
        nan_mask_X2s = torch.isnan(X2s).any(dim=(2, 3))
        nan_mask_X2s_broker = torch.isnan(X2s_broker).any(dim=(2, 3))
        nan_mask_Ys = torch.isnan(Ys)
        valid_mask = (~(nan_mask_Xs | nan_mask_Xs_broker | nan_mask_X2s | nan_mask_X2s_broker | nan_mask_Ys)).float()
        return self.criterion(preds, Ys, valid_mask)

    def train_with_early_stop(self):
        """
        First phase: train on train_loader, evaluate on val_loader, use early stopping.
        Returns optimal epoch count (excluding patience)."""
        early_stop = EarlyStop(
            patience=self.args.patience, min_delta=0.01
        )
        num_epochs = self.args.train_epochs
        best_epoch = num_epochs
        for epoch in range(1, num_epochs + 1):
            self._train_one_epoch(epoch, self.train_loader)
            val_loss = self.evaluate(self.val_loader)
            print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f}")
            early_stop(val_loss)
            if early_stop.stop_training:
                print(f"Early stopping at epoch {epoch}")
                best_epoch = epoch - self.args.patience
                break
        return best_epoch

    def _train_one_epoch(self, epoch, loader):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            enumerate(loader), 
            total=len(loader), 
            desc=f"Epoch {epoch}",
            leave=False
        )

        for batch_idx, (Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys, _) in progress_bar:
            Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys = self._prepare_batch(
                Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys
            )
            self.optimizer.zero_grad()
            preds = self.model.forward_train(Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general)
            # breakpoint()
            loss = self._masked_loss(preds, Ys, Xs, Xs_broker, X2s, X2s_broker)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        print(
            f"Epoch {epoch} completed. Avg Loss: {total_loss / len(loader):.4f}"
        )

    def evaluate(self, loader):
        """Evaluate model on a loader."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys, _ in loader:
                Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys = self._prepare_batch(
                    Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys
                )
                preds = self.model.forward_train(Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general)
                total_loss += self._masked_loss(
                    preds, Ys, Xs, Xs_broker, X2s, X2s_broker
                ).item()
        return total_loss / len(loader)

    def _reinitialize_model(self):
        def weights_init(m):
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        torch.nn.init.zeros_(param)
        self.model.apply(weights_init)
        
    def train_final(self, num_epochs):
        """
        Second phase: retrain model on combined train+val data.
        num_epochs should be best_epoch from first phase."""
        print("Reinitializing model before final training...")
        self._reinitialize_model()
        
        print(f"Retraining on train+val for {num_epochs} epochs")
        for epoch in range(1, num_epochs + 1):
            self._train_one_epoch(epoch, self.train_val_loader)
        print("Final training completed.")

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def save_args(self, path):
        """
        儲存 args 為 JSON 檔案。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(os.path.join(path, "preprocessor_args.json"), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)

    def _evaluate_and_save(
        self,
        split_name: str,
        dataset,
        loader,
        split_dir: str
    ):
        """
        在已經計算好的 split_dir 下，對單一 split 做 inference 並存檔。
        """
        os.makedirs(split_dir, exist_ok=True)

        # 1) 儲存 args.json
        with open(os.path.join(split_dir, "preprocessor_args.json"), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)

        # 2) 預測並收集
        all_pred, all_true, all_idx = [], [], []
        all_pred_pct, all_true_pct = [], []

        self.model.eval()
        with torch.no_grad():
            for Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys, indices in loader:
                Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys = self._prepare_batch(
                    Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general, Ys
                )
                preds = self.model.forward_train(Xs, Xs_broker, Xs_general, X2s, X2s_broker, X2s_general)

                # 避免 NAN propagate
                mask = ~(
                    torch.isnan(Xs).any(dim=(2,3)) |
                    torch.isnan(X2s).any(dim=(2,3)) |
                    torch.isnan(Ys)
                )
                preds = preds.masked_fill(~mask, float("nan"))

                preds_np = preds.cpu().numpy()
                Ys_np    = Ys.cpu().numpy()
                idx_np   = indices.cpu().numpy()

                all_pred.append(preds_np)
                all_true.append(Ys_np)
                all_idx.append(idx_np)
                all_pred_pct.append([
                    dataset.get_pct(p, i) for p, i in zip(preds_np, idx_np)
                ])
                all_true_pct.append([
                    dataset.get_pct(y, i) for y, i in zip(Ys_np, idx_np)
                ])

        # 3) 合併矩陣
        pred_mat    = np.concatenate(all_pred,     axis=0)
        true_mat    = np.concatenate(all_true,     axis=0)
        idx_mat     = np.concatenate(all_idx,      axis=0)
        pred_pct    = np.concatenate(all_pred_pct, axis=0)
        true_pct    = np.concatenate(all_true_pct, axis=0)

        # 4) 構造 DataFrame 並存 CSV
        dates = np.array([dataset.get_date(i) for i in idx_mat]).reshape(-1, 1)
        cols  = ["date"] + dataset.get_stock_ids()

        def _save_csv(name: str, mat: np.ndarray):
            df = pd.DataFrame(np.hstack([dates, mat]), columns=cols)
            df.to_csv(os.path.join(split_dir, f"{name}.csv"), index=False)

        _save_csv("pred",     pred_mat)
        _save_csv("truth",    true_mat)
        _save_csv("pred_pct", pred_pct)
        _save_csv("truth_pct", true_pct)

        print(f"[INFO] Saved '{split_name}' results to {split_dir}")

    def evaluate_and_save(self, base_output_dir: str):
        """
        1) 用 test_set 的日期計算 timespan；
        2) 在 {base_output_dir}/{timespan}/preprocessor/ 下，分別跑 test & train_val；
        """
        base_output_dir = os.path.join(base_output_dir, "preprocessor")
        # --- 計算 test timespan  ---
        start, end = pd.to_datetime(self.args.split_dates[2]), pd.to_datetime(self.args.split_dates[3])
        timespan = f"{start:%Y%m%d}_{end:%Y%m%d}"

        # --- 為兩個 split 準備各自資料夾 ---
        root_dir = os.path.join(base_output_dir, timespan)
        test_dir = os.path.join(root_dir, "test")
        tv_dir   = os.path.join(root_dir, "train_val")

        # --- 執行並存檔 ---
        self._evaluate_and_save(
            split_name="test",
            dataset=self.test_set,
            loader=self.test_loader,
            split_dir=test_dir
        )
        self._evaluate_and_save(
            split_name="train_val",
            dataset=self.train_val_set,
            loader=self.train_val_loader,
            split_dir=tv_dir
        )
