"""Data transformation module for classification."""

import pandas as pd
import os
import numpy as np
import tqdm
import shutil

class RegressionSplitResult:
    """Sub-Class to handle regression result conversion between trading-pipeline and trading-model in a split.
    
    It will transform the data to a unified format for both trading-pipeline and trading-model:

    date: str, stock_id: str, pred_pct: float, truth_pct: float,

    Properties:
    - data: pd.DataFrame - the data in unified format
    """

    data: pd.DataFrame = None # in unified format

    @staticmethod
    def from_pipeline(from_path: str):
        # print(f"Loading data from {from_path}...")

        # load data
        pred_df = pd.read_csv(os.path.join(from_path, 'pred_pct.csv'))
        truth_df = pd.read_csv(os.path.join(from_path, 'truth_pct.csv'))
        
        melted_pred_df = pred_df.melt(
            id_vars=["date"],
            var_name="stock_id",
            value_name="pred_pct"
        )
        melted_truth_df = truth_df.melt(
            id_vars=["date"],
            var_name="stock_id",
            value_name="truth_pct"
        )

        # merge pred_logits and truth
        final_df = melted_pred_df.merge(melted_truth_df, on=["date", "stock_id"])
        final_df['stock_id'] = final_df['stock_id'].astype(str)
        final_df = final_df[["date", "stock_id", "pred_pct", "truth_pct"]]
        
        self = RegressionSplitResult()
        self.data = final_df

        return self
        
    @staticmethod
    def from_model(from_path: str):
        # print(f"Loading data from {from_path}...")

        # Load data
        file_path = os.path.join(from_path, "whole_output.csv")
        df = pd.read_csv(file_path)

        # Rename and transform columns
        df.rename(columns={'true_pct': 'truth_pct'}, inplace=True)
        df['stock_id'] = df['stock_id'].astype(str)

        self = RegressionSplitResult()
        self.data = df[["date", "stock_id", "pred_pct", "truth_pct"]]

        return self

    def to_pipeline(self, to_path: str):
        final_df = self.data
        stock_id_cols = sorted(final_df['stock_id'].unique().astype(str).tolist())
        
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
        os.makedirs(to_path, exist_ok=True)

        # pivot and set NaN as prediction 0.
        pred_df = final_df.pivot(index="date", columns="stock_id", values="pred_pct").reset_index()
        pred_df[stock_id_cols] = pred_df[stock_id_cols].fillna(0)
        pred_df.to_csv(os.path.join(to_path, 'pred_pct.csv'), index=False)

        # truth.csv (NaN is labeled as -1)
        truth_df = final_df.pivot(index="date", columns="stock_id", values="truth_pct").reset_index()
        truth_df[stock_id_cols] = truth_df[stock_id_cols].fillna(0)
        truth_df.to_csv(os.path.join(to_path, 'truth_pct.csv'), index=False)

        # print(f"Transformed data saved to {to_path}/pred_pct.csv, truth_pct.csv")

    def to_model(self, to_path: str):
        final_df = self.data

        if os.path.exists(to_path):
            shutil.rmtree(to_path)
        os.makedirs(to_path, exist_ok=True)

        # Transform to model format
        model_df = final_df[['stock_id', 'date', 'pred_pct', 'truth_pct']].copy()
        model_df.rename(columns={'truth_pct': 'true_pct', 'pred_pct': 'pred_pct'}, inplace=True)

        # Save to CSV
        model_df.to_csv(os.path.join(to_path, 'whole_output.csv'), index=False)
        # print(f"Transformed data saved to {to_path}/whole_output.csv")

class RegressionResult:
    """Class to handle regression result transformation between trading-pipeline and trading-model.
    
    Properties:
    - train_val_data: RegressionSplitResult - the training and validation split result
    - test_data: RegressionSplitResult - the test split result
    """
    train_val_data: RegressionSplitResult = None
    test_data: RegressionSplitResult = None

    @staticmethod
    def from_pipeline(from_path: str):
        self = RegressionResult()
        self.train_val_data = RegressionSplitResult.from_pipeline(os.path.join(from_path, 'train_val'))
        self.test_data = RegressionSplitResult.from_pipeline(os.path.join(from_path, 'test'))
        return self

    @staticmethod
    def from_model(from_path: str):
        self = RegressionResult()
        self.train_val_data = RegressionSplitResult.from_model(os.path.join(from_path, 'train_vali'))
        self.test_data = RegressionSplitResult.from_model(os.path.join(from_path, 'test'))
        return self

    def to_pipeline(self, to_path):
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
        os.makedirs(to_path, exist_ok=True)
        self.train_val_data.to_pipeline(os.path.join(to_path, 'train_val'))
        self.test_data.to_pipeline(os.path.join(to_path, 'test'))

    def to_model(self, to_path):
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
        os.makedirs(to_path, exist_ok=True)
        self.train_val_data.to_model(os.path.join(to_path, 'train_vali'))
        self.test_data.to_model(os.path.join(to_path, 'test'))