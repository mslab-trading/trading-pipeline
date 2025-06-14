import pandas as pd
import os

def read_all_df(result_dir, start_year=2021, end_year=2024):
    test_years = list(range(start_year, end_year + 1))
    
    all_pred = pd.DataFrame()
    all_truth = pd.DataFrame()
    all_pred_val = pd.DataFrame()
    all_truth_val = pd.DataFrame()

    for test_year in test_years:
        split = f"{test_year}"
        val_year = test_year - 1
        
        pred_file = os.path.join(result_dir, split, "test", "pred_pct.csv")
        truth_file = os.path.join(result_dir, split, "test", "truth_pct.csv")
        pred_train_val_file = os.path.join(result_dir, split, "train_val", "pred_pct.csv")
        truth_train_val_file = os.path.join(result_dir, split, "train_val", "truth_pct.csv")
        
        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        # 讀入當前 split 的預測 CSV
        def read_df(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.set_index('date', inplace=True)
            df = df.sort_index()
            return df
        
        pred_df = read_df(pred_file)
        truth_df = read_df(truth_file)
        
        pred_train_val_df = read_df(pred_train_val_file)
        truth_train_val_df = read_df(truth_train_val_file)
        
        pred_val_df = pred_train_val_df[pred_train_val_df.index.year == val_year]
        truth_val_df = truth_train_val_df[truth_train_val_df.index.year == val_year]
        

        # 將讀到的資料串接到 all_pred
        all_pred = pd.concat([all_pred, pred_df], axis=0)
        all_truth = pd.concat([all_truth, truth_df], axis=0)
        all_pred_val = pd.concat([all_pred_val, pred_val_df], axis=0)
        all_truth_val = pd.concat([all_truth_val, truth_val_df], axis=0)

    # 針對同一個 date，保留最後出現（即後面）的資料
    all_pred = all_pred.groupby(level=0).last()
    all_truth = all_truth.groupby(level=0).last()
    return all_pred, all_truth, all_pred_val, all_truth_val