import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Any
from collections import defaultdict
import json
import random

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

class Dataset_Basic(Dataset):
    def __init__(
        self,
        market_df,
        broker_df,
        global_df,
        size,
        flag,
        target,
        split_dates,
    ):
        """
        A Dataset class for a time-series forecasting task.
        ...
        """
        super().__init__()

        # Parse size tuple
        self.seq_len, self.pred_len = size

        # Basic assignments
        self.flag = flag
        self.target = target
        self.split_dates = [pd.to_datetime(sd) for sd in split_dates]

        # Filter out rare stocks
        self.stock_ids = self.filter_stock_ids(market_df)
        self.stock_ids = sorted(self.stock_ids)

        market_df = market_df[market_df["stock_id"].isin(self.stock_ids)].copy()
        broker_df = broker_df[broker_df["stock_id"].isin(self.stock_ids)].copy()

        self.all_dates = sorted(market_df["date"].drop_duplicates().tolist())
        self.split_dates = self.adjust_split_dates(split_dates)
        
        # Extract data based on the flag
        extracted_market_df = self.extract_data(market_df, self.flag)
        extracted_broker_df = self.extract_data(broker_df, self.flag)

        # Sort by date 
        extracted_market_df = extracted_market_df.sort_values(by="date")
        extracted_broker_df = extracted_broker_df.sort_values(by="date")

        
        # Store final DataFrame and unique dates
        self.market_df = extracted_market_df
        self.broker_df = extracted_broker_df
        self.dates = extracted_market_df["date"].sort_values().unique()
        
        # Read global data
        self.global_df = global_df.copy()
        
    def __len__(self):
        # The number of usable entries is reduced by seq_len + pred_len
        return len(self.dates) - self.pred_len - self.seq_len

    def __getitem__(self, index):
        """
        Raise NotImplementedError for now.
        """
        raise NotImplementedError("`__getitem__` method is not implemented yet.")
    
    def adjust_split_dates(self, raw_split_dates):
        """
        只修改 raw_split_dates 的第 2、3 個元素（索引 1、2）──
        把它們往前退 self.seq_len 個交易日，
        其餘邊界保持原始 calendar date。
        
        參數
        ----
        raw_split_dates: list of str or Timestamp, 長度必須為 4
            例如 ["2018-01-01","2020-01-01","2021-01-01","2022-01-01"]

        回傳
        ----
        List[pd.Timestamp] 長度 4
        """
        # 1) 轉成 Timestamp
        raw = [pd.to_datetime(d) for d in raw_split_dates]
        assert len(raw) == 4, "需要 4 個 split date"

        # 2) 確保 self.all_dates 已經建立（排序過的交易日列表）
        assert hasattr(self, "all_dates"), "請先設定 self.all_dates"
        
        # 3) 新的 split list，先把第 0 個放進去
        new_splits = []

        # 4) 只對索引 1 和 2 做交易日退後
        for i in (0, 1, 2):
            # 找到 all_dates 中第一個 ≥ raw[i] 的位置 j
            j = next(idx for idx, x in enumerate(self.all_dates) if x >= raw[i])
            # 退後 seq_len 個交易日（若不足就取 0）
            shifted_idx = max(0, j - self.seq_len)
            new_splits.append(self.all_dates[shifted_idx])

        
        j3 = next(idx for idx, x in enumerate(self.all_dates) if x >= raw[3])
        # 往後推 pred_len，最多到列表末尾
        shifted_idx3 = min(len(self.all_dates) - 1, j3 + self.pred_len)
        new_splits.append(self.all_dates[shifted_idx3])
        
        return new_splits

    def filter_stock_ids(self, df, threshold=0.9):
        """
        Keep only stock_ids that have at least 'threshold'% of the maximum presence.
        Also print the stock_ids that fall below this threshold.
        """
        stock_id_counts = df["stock_id"].value_counts()
        max_count = stock_id_counts.max()
        threshold_value = max_count * threshold

        # Separate IDs that pass vs. fail the threshold
        passed_ids = []
        below_ids = []
        for sid, count in stock_id_counts.items():
            if count > threshold_value:
                passed_ids.append(sid)
            else:
                below_ids.append(sid)

        # Print or log the stock_ids that are below threshold
        if below_ids:
            print(f"Stock IDs below threshold ({threshold:.2f}): {below_ids}")
        else:
            print("No Stock IDs below threshold")

        return passed_ids

    def extract_data(self, df, flag):
        """
        Slice df by date range based on 'flag' and 'self.split_dates'.
        """
        if flag == "train":
            left, right = self.split_dates[0], self.split_dates[1]
        elif flag == "val":
            left, right = self.split_dates[1], self.split_dates[2]
        elif flag == "test":
            left, right = self.split_dates[2], self.split_dates[3]
        elif flag == "train_val":
            left, right = self.split_dates[0], self.split_dates[2]
        else:
            raise ValueError(f"Unknown flag: {flag}. Must be train/val/test/train_val.")

        left, right = pd.Timestamp(left), pd.Timestamp(right)
        try:
            mask = (df["date"] >= left) & (df["date"] < right)
        except:
            breakpoint()
        return df[mask]

    def get_base_item(self, index):
        seq_df, pred_df = self._get_base_item_df(index, self.market_df)
        broker_seq_df, broker_pred_df = self._get_base_item_df(index, self.broker_df)
        broker_seq_df = broker_seq_df.reindex(seq_df.index, fill_value=0)
        global_seq_df = self.global_df.reindex(seq_df.index).ffill().fillna(0)
        
        return seq_df, broker_seq_df, global_seq_df, pred_df 
        
    def _get_base_item_df(self, index, df):
        """
        Construct seq_df and pred_df by pivoting around a specific date index range.
        Returns scaled DataFrames.
        ...
        """
        # Shift index so that 'index' points to the forecast anchor.
        index += self.seq_len  
        
        # Check that we can still get self.pred_len days into the future
        if index + self.pred_len > len(self.dates):
            raise IndexError(f"Index {index} out of date range {len(self.dates)}")
        
        # Identify start/end points in the self.dates array
        s_start = self.dates[index - self.seq_len]
        s_end   = self.dates[index]
        r_start = self.dates[index]           # forecast starts exactly where seq ends
        r_end   = self.dates[index + self.pred_len]
        
        # Filter the entire (seq + pred) window at once
        crr_df = df[(df["date"] >= s_start) & (df["date"] < r_end)]
        
        # Pivot so rows = date, columns = (stock_id, feature)
        pivot_df = crr_df.pivot(index="date", columns="stock_id")
        pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
        
        # 3) Build a full set of columns for reindexing:
        all_features = pivot_df.columns.levels[1]   # The features that do exist
        # If pivot_df.columns.names is something like ['stock_id', None], keep that ordering:
        all_cols = pd.MultiIndex.from_product(
            [self.stock_ids, all_features],
            names=pivot_df.columns.names
        )

        # 4) Reindex so that pivot_df has all stocks and all features
        pivot_df = pivot_df.reindex(columns=all_cols)
        
        # Slice out the sequence portion: [s_start, s_end)
        seq_df  = pivot_df.loc[(pivot_df.index >= s_start) & (pivot_df.index < s_end)]
        
        # Slice out the prediction portion: [r_start, r_end)
        pred_df = pivot_df.loc[(pivot_df.index >= r_start) & (pivot_df.index < r_end)]
        
        return seq_df, pred_df

    def get_num_stocks(self):
        """
        Return the number of unique stock_ids in the dataset.
        """
        return len(self.stock_ids)

    def get_stock_ids(self):
        """
        Return the list of unique stock_ids in the dataset.
        """
        return self.stock_ids
    
    def get_date(self, index):
        return self.dates[index + self.seq_len]
    

class Dataset_Abs(Dataset_Basic):
    def __init__(self, *args, goal, log, thresh = None, scaler_type = 'standard', **kwargs):
        """
        A specialized Dataset class extending Dataset_Basic, focusing on a 
        particular 'goal' (e.g., 'max_price' or 'last_price') of the target variable.
        
        Parameters
        ----------
        goal : str
            Specifies how to extract the label from the prediction window.
            Must be either 'max_price' or 'last_price'.
        *args, **kwargs : 
            Passed through to the parent Dataset_Basic constructor.
        """
        super().__init__(*args, **kwargs)
        self.goal = goal
        self.log = log
        self.thresh = thresh
        
        if scaler_type == 'min_max':
            self.scaler_class = MinMaxScaler
        elif scaler_type == 'standard':
            self.scaler_class = StandardScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Must be 'min_max' or 'standard'.")
    
    def scale_data(self, seq_df, pred_df = None):
        """
        Scale seq_df and pred_df with the given scaler type (MinMax or Standard).
        
        Parameters
        ----------
        seq_df : pd.DataFrame
            Sequence window data to be scaled (rows: dates, MultiIndex columns: (stock_id, feature)).
        pred_df : pd.DataFrame
            Prediction window data to be scaled.
        scaler_type : str, optional
            'min_max' or 'standard' (default: 'min_max').
        
        Returns
        -------
        seq_df_scaled, pred_df_scaled : tuple of pd.DataFrame
            Scaled versions of seq_df and pred_df, matching their original shapes/index/columns.
        """
        scaler = self.scaler_class()

        # Fit the scaler on the seq_df portion
        scaler.fit(seq_df)
        
        def scale_df(dframe, fitted_scaler):
            arr = fitted_scaler.transform(dframe)
            return pd.DataFrame(arr, index=dframe.index, columns=dframe.columns)

        # Apply transformation to both sequence and prediction data
        seq_df_scaled = scale_df(seq_df, scaler)
        
        if pred_df is None:
            return seq_df_scaled
        
        pred_df_scaled = scale_df(pred_df, scaler)
        return seq_df_scaled, pred_df_scaled
    
    def reshape_2d_to_3d(self, seq_df):
        """
        Reshape a pivoted DataFrame (T × (N×F)) into a 3D NumPy array (N × T × F).
        
        Parameters
        ----------
        seq_df : pd.DataFrame
            Rows = T time steps, MultiIndex columns = (N stocks, F features).
        
        Returns
        -------
        arr_3d : np.ndarray
            A 3D array with shape (N, T, F).
        """
        # Convert to (T, N*F) NumPy array
        arr_2d = seq_df.to_numpy()
        
        # Determine dimensions: N = # unique stock_ids, F = # features, T = # time steps
        N = seq_df.columns.levels[0].size
        F = seq_df.columns.levels[1].size
        T = seq_df.shape[0]

        # Reshape to (T, N, F) then transpose to (N, T, F)
        arr_3d = arr_2d.reshape(T, N, F).transpose(1, 0, 2)
        return arr_3d

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        For the given 'index', get the sequence and prediction DataFrames, scale them,
        and then extract the label 'Y' based on 'self.goal':
          - 'max_price': maximum of the target column over the prediction window
          - 'last_price': last value of the target column in the prediction window
        
        Returns
        -------
        X : np.ndarray
            A 3D array of shape (N, T, F) representing scaled sequence data.
        Y : np.ndarray
            1D array of length N with either max or last target price (per stock).
        index : int
            The same index provided, for tracking purposes.
        """
        # Fetch the unscaled seq and pred DataFrames for this index
        seq_df, seq_broker_df, seq_global_df, pred_df = self.get_base_item(index)
        
        # create a mask for the stocks that are above the threshold
        if self.thresh:
            close_seq_df = seq_df.xs(self.target, axis=1, level=1)
            thresh_df = close_seq_df.iloc[-1:] * (1 + self.thresh)
            close_seq_df, thresh_df = self.scale_data(close_seq_df, thresh_df)
            
        # Scale data
        seq_df, pred_df = self.scale_data(seq_df, pred_df)
        seq_broker_df = self.scale_data(seq_broker_df)
        seq_global_df = self.scale_data(seq_global_df)
        
        # Extract only the target column from pred_df
        # This returns a DataFrame with shape (T_pred, N_stocks)
        close_pred_df = pred_df.xs(self.target, axis=1, level=1)
        

        # Compute the label Y based on goal
        if self.goal == 'max_price':
            # Max over the prediction window for each stock
            Y = close_pred_df.max(skipna=True).values  # shape: (N_stocks,)
        elif self.goal == 'min_price':
            # Max over the prediction window for each stock
            Y = close_pred_df.min(skipna=True).values  # shape: (N_stocks,)   
        elif self.goal == 'mean_price':
            # Mean over the prediction window for each stock
            Y = close_pred_df.mean(skipna=True).values  # shape: (N_stocks,)
        elif self.goal == 'last_price':
            # Last value in the prediction window for each stock
            Y = close_pred_df.iloc[-1].values
        else:
            raise ValueError(f"goal = {self.goal} not implemented yet, only max_price, min_price, mean_price are supported")

        # Convert the sequence df into a 3D array
        X = self.reshape_2d_to_3d(seq_df)
        seq_broker_df = seq_broker_df.fillna(0)
        X_broker = self.reshape_2d_to_3d(seq_broker_df)
        
        # if np.isnan(X).sum() > 100:
        #     breakpoint()
        
        if self.log:
            # Clip Y to a tiny positive constant if any are <= 0
            Y = np.clip(Y, 1e-8, None)
            Y = np.log(Y)  # log transform

        if self.thresh:
            return X, X_broker, Y, thresh_df.values.flatten(), index
        
        return X, X_broker, seq_global_df.values, Y, index

    def inverse(self, pred_y, index):
        """
        Inverse the scaling on a predicted target array, 'pred_y'.
        
        Parameters
        ----------
        pred_y : np.ndarray
            A 1D array of scaled predictions for each stock (shape: (N_stocks,)).
        index : int
            The dataset index for which to fetch the original (unscaled) sequence data.
        
        Returns
        -------
        np.ndarray
            The same shape as pred_y, but with scaling reversed to original price scale.
        """
        if self.log:
            pred_y = np.exp(pred_y)
        # Retrieve the original, unscaled sequence + pred data
        seq_df, broker_seq_df, global_seq_df, pred_df = self.get_base_item(index)
        
        # We'll fit a MinMaxScaler only on the target column in seq_df
        # so we can invert 'pred_y' properly.
        seq_df_close_only = seq_df.xs(self.target, axis=1, level=1)
        
        scaler = self.scaler_class()
        scaler.fit(seq_df_close_only)
        
        # Reshape pred_y to (1, N_stocks) so inverse_transform works
        pred_y_2d = pred_y.reshape(1, -1)
        pred_y_inversed = scaler.inverse_transform(pred_y_2d)
        
        # Flatten back to 1D
        return pred_y_inversed.ravel()

    def get_pct(self, pred_y, index):
        """
        Calculate the percentage change between the last price in the sequence
        and the predicted price for each stock.
        
        Parameters
        ----------
        pred_y : np.ndarray
            A 1D array of predicted prices for each stock (shape: (N_stocks,)).
        index : int
            The dataset index for which to fetch the original (unscaled) sequence data.
        
        Returns
        -------
        np.ndarray
            A 1D array of percentage changes for each stock (shape: (N_stocks,)).
        """
        # Retrieve the original, unscaled sequence + pred data
        seq_df, seq_broker_df, seq_global_df, pred_df = self.get_base_item(index)
        
        # Extract the last price in the sequence for each stock
        seq_close = seq_df.xs(self.target, axis=1, level=1).iloc[-1].values
        
        price_y = self.inverse(pred_y, index)
        
        # Calculate the percentage change
        return (price_y - seq_close) / seq_close 
    
class Dataset_Pct(Dataset_Basic):
    def __init__(self, *args, goal, log, thresh = None, scaler_type = 'standard', **kwargs):
        """
        A specialized Dataset class extending Dataset_Basic, focusing on a 
        particular 'goal' (e.g., 'max_price' or 'last_price') of the target variable.
        
        Parameters
        ----------
        goal : str
            Specifies how to extract the label from the prediction window.
            Must be either 'max_price' or 'last_price'.
        *args, **kwargs : 
            Passed through to the parent Dataset_Basic constructor.
        """
        super().__init__(*args, **kwargs)
        self.goal = goal
        self.log = log
        self.thresh = thresh
        
        if scaler_type == 'min_max':
            self.scaler_class = MinMaxScaler
        elif scaler_type == 'standard':
            self.scaler_class = StandardScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Must be 'min_max' or 'standard'.")
    
    def scale_data(self, seq_df, pred_df = None):
        """
        Scale seq_df and pred_df with the given scaler type (MinMax or Standard).
        
        Parameters
        ----------
        seq_df : pd.DataFrame
            Sequence window data to be scaled (rows: dates, MultiIndex columns: (stock_id, feature)).
        pred_df : pd.DataFrame
            Prediction window data to be scaled.
        scaler_type : str, optional
            'min_max' or 'standard' (default: 'min_max').
        
        Returns
        -------
        seq_df_scaled, pred_df_scaled : tuple of pd.DataFrame
            Scaled versions of seq_df and pred_df, matching their original shapes/index/columns.
        """
        scaler = self.scaler_class()

        # Fit the scaler on the seq_df portion
        scaler.fit(seq_df)
        
        def scale_df(dframe, fitted_scaler):
            arr = fitted_scaler.transform(dframe)
            return pd.DataFrame(arr, index=dframe.index, columns=dframe.columns)

        # Apply transformation to both sequence and prediction data
        seq_df_scaled = scale_df(seq_df, scaler)
        
        if pred_df is None:
            return seq_df_scaled
        
        pred_df_scaled = scale_df(pred_df, scaler)
        return seq_df_scaled, pred_df_scaled
    
    def reshape_2d_to_3d(self, seq_df):
        """
        Reshape a pivoted DataFrame (T × (N×F)) into a 3D NumPy array (N × T × F).
        
        Parameters
        ----------
        seq_df : pd.DataFrame
            Rows = T time steps, MultiIndex columns = (N stocks, F features).
        
        Returns
        -------
        arr_3d : np.ndarray
            A 3D array with shape (N, T, F).
        """
        # Convert to (T, N*F) NumPy array
        arr_2d = seq_df.to_numpy()
        
        # Determine dimensions: N = # unique stock_ids, F = # features, T = # time steps
        N = seq_df.columns.levels[0].size
        F = seq_df.columns.levels[1].size
        T = seq_df.shape[0]

        # Reshape to (T, N, F) then transpose to (N, T, F)
        arr_3d = arr_2d.reshape(T, N, F).transpose(1, 0, 2)
        return arr_3d

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        For the given 'index', get the sequence and prediction DataFrames, scale them,
        and then extract the label 'Y' based on 'self.goal':
          - 'max_price': maximum of the target column over the prediction window
          - 'last_price': last value of the target column in the prediction window
        
        Returns
        -------
        X : np.ndarray
            A 3D array of shape (N, T, F) representing scaled sequence data.
        Y : np.ndarray
            1D array of length N with either max or last target price (per stock).
        index : int
            The same index provided, for tracking purposes.
        """
        # Fetch the unscaled seq and pred DataFrames for this index
        seq_df, seq_broker_df, seq_global_df, pred_df = self.get_base_item(index)
        
        # Extract only the target column from pred_df
        # This returns a DataFrame with shape (T_pred, N_stocks)
        close_pred_df = pred_df.xs(self.target, axis=1, level=1)
        first_close_pred = close_pred_df.iloc[0].values.flatten()
        
        # Compute the label Y based on goal
        if self.goal == 'max_roi':
            # Max over the prediction window for each stock
            Y = close_pred_df.max(skipna=True).values/first_close_pred - 1 # shape: (N_stocks,)
        elif self.goal == 'min_roi':
            # Max over the prediction window for each stock
            Y = close_pred_df.min(skipna=True).values/first_close_pred - 1# shape: (N_stocks,)   
        elif self.goal == 'mean_roi':
            # Mean over the prediction window for each stock
            Y = close_pred_df.mean(skipna=True).values/first_close_pred - 1# shape: (N_stocks,)
        elif self.goal == 'last_roi':
            # Last value of the target column in the prediction window for each stock
            Y = close_pred_df.iloc[-1].values/first_close_pred - 1
        else:
            raise ValueError(f"goal = {self.goal} not implemented yet, only max_roi, min_roi, mean_roi are supported")
        
        # Scale data
        seq_df = self.scale_data(seq_df)
        seq_broker_df = self.scale_data(seq_broker_df)
        seq_global_df = self.scale_data(seq_global_df)
        
        # Convert the sequence df into a 3D array
        X = self.reshape_2d_to_3d(seq_df)
        seq_broker_df = seq_broker_df.fillna(0)
        X_broker = self.reshape_2d_to_3d(seq_broker_df)
        
        # if np.isnan(X).sum() > 100:
        #     breakpoint()
        
        if self.log:
            # Clip Y to a tiny positive constant if any are <= 0
            Y = np.clip(Y, 1e-8, None)
            Y = np.log(Y)  # log transform
            
        return X, X_broker, seq_global_df.values, Y, index

    def inverse(self, pred_y, index):
        return pred_y

    def get_pct(self, pred_y, index):
        return pred_y 

class Dataset_S3E(Dataset_Basic):
    def __init__(self, *args, data="Dataset_Abs", **kwargs):
        # work like data_factory
        if data == "Dataset_Pct":
            self.dataset = Dataset_Pct(*args, **kwargs)
        elif data == "Dataset_Abs":
            self.dataset = Dataset_Abs(*args, **kwargs)
        else:
            raise ValueError(f"Unknown data type: {data}. Must be 'Dataset_Pct' or 'Dataset_Abs' in Dataset_S3E.")

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.
        For the given 'index', get the sequence and prediction DataFrames, scale them,
        and then extract the label 'Y' based on 'self.goal':
          - 'max_price': maximum of the target column over the prediction window
          - 'last_price': last value of the target column in the prediction window
        Returns
        -------
        X : np.ndarray
            A 3D array of shape (N, T, F) representing 1st sequence data.
        X2 : np.ndarray
            A 3D array of shape (N, T, F) representing 2nd sequence data.
        Y : np.ndarray
            1D array of length N with the target value, which is the difference between the last price in X2 and the last price in X.
        index : int
            The same index provided, for tracking purposes.
        -----------
        """
        X, X_broker, X_global, Y, index = self.dataset[index] # 1st sequence data
        index2 = random.choice([i for i in range(len(self.dataset)) if i != index])
        X2, X2_broker, X2_global, Y2, _ = self.dataset[index2]  # 2nd sequence data

        Y = Y - Y2  # Target value is the difference between the last price in X2 and the last price in X

        return X, X_broker, X_global, X2, X2_broker, X2_global, Y, index

    def inverse(self, pred_y, index):
        return pred_y

    def get_pct(self, pred_y, index):
        return pred_y

# TODO: Jerome
class Dataset_Jerome(Dataset_Basic):
    def __init__(self):
        pass
    
# TODO: Berlin
class Dataset_Berlin(Dataset_Basic):
    def __init__(self):
        pass

if __name__ == "__main__":
    print("data loader testing")
    root_path = "./data/raw/market/"
    broker_path = "./data/raw/broker/"
    global_data_path = "./data/raw/global/global_data.csv"
    
    stock_ids =  ['2330', '2317', '2454']  # Example stock IDs
    market_features = ['etl:adj_close','etl:adj_open','etl:adj_high','etl:adj_low','price:成交筆數']
    global_features = ["^VIX", "PCR_Volume"]
    
    size = (60, 30)
    
    
    from data_reader import (
        read_market_data,
        read_global_data,
        read_broker_data,
    )
        
    market_df = read_market_data(
        root_path,
        stock_ids,
        global_data_path=global_data_path,
        market_features=market_features,
        global_features=global_features
    )

    broker_df = read_broker_data(
        broker_path,
        stock_ids
    )
    
    global_df = read_global_data(
        global_data_path, 
        global_features=global_features
    )
    
    dataset_basic = Dataset_Basic(
        market_df=market_df,
        broker_df=broker_df,
        global_df=global_df,
        size=size,
        flag='test', 
        target="etl:adj_close", 
        split_dates=["2020-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
    )

    seq_df, seq_broker_df, seq_global_df, pred_df = dataset_basic.get_base_item(0)
    seq_broker_shape = seq_broker_df.shape
    
    for i in range(len(dataset_basic)):
        seq_df, seq_broker_df, seq_global_df, pred_df = dataset_basic.get_base_item(i)
        if seq_broker_df.shape != seq_broker_shape:
            print(f"Shape mismatch at index {i}: {seq_broker_shape}")
    
    
    dataset_abs = Dataset_Abs(
        market_df=market_df,
        broker_df=broker_df,
        global_df=global_df,
        size=size,
        flag='train', 
        target="etl:adj_close", 
        split_dates=["2018-01-01", "2025-01-01", "2025-01-01", "2025-01-01"],
        goal='max_price',
        log=0,
    )
    
    X_shape, X_broker_shape, X_global_shape, Y_shape = dataset_abs[0][0].shape, dataset_abs[0][1].shape, dataset_abs[0][2].shape, dataset_abs[0][3].shape
    

    for X, X_broker, X_global, Y, index in dataset_abs:
        if X.shape != X_shape:
            print(f"Shape mismatch at index {index}: {X_shape}")
        if X_broker.shape != X_broker_shape:
            print(f"Shape mismatch at index {index}: {X_broker_shape}")
        if X_global.shape != X_global_shape:
            print(f"Shape mismatch at index {index}: {X_global_shape}")
        if Y.shape != Y_shape:
            print(f"Shape mismatch at index {index}: {Y_shape}")
        dataset_abs.get_pct(Y, index)
        
    # if True:
    #     for X, Y, thresh, index in dataset_abs:
    #         pass
            # _, pred_df = dataset_abs.get_base_item(index)
            # original = pred_df.xs("etl:adj_close", axis=1, level=1).max().values
            # inversed = dataset_abs.inverse(Y, index)

            # # If these are floating-point arrays, better to compare with a tolerance:
            # diff_mask = ~np.isclose(original, inversed, rtol=1e-5, atol=1e-8)

            # if False and sum(diff_mask) > 0:
            #     print("Index:", index)
            #     print("Positions where they differ:", np.where(diff_mask)[0])
            #     print("Original values at those positions:", original[diff_mask])
            #     print("Inversed values at those positions:", inversed[diff_mask])

            # if X.shape != X0.shape or Y.shape != Y0.shape:
            #     print(f"Shape mismatch at index: {index}, X: {X.shape}, Y: {Y.shape}")
                
            # if False and np.isnan(X).any() or np.isnan(Y).any():
            #     print(f"NaN detected at index {index}!")
            #     print(f"X contains NaN: {np.isnan(X).sum()} values")
            #     print(f"Y contains NaN: {np.isnan(Y).sum()} values")
            #     # breakpoint()
    
    # if False:
    #     for X, Y, index in dataset_abs:
    #         print(f"Index: {index}, Date: {dataset_abs.get_date(index)}")
    
    # for X, Y, index in dataset_abs:
    #     reversed_pct = dataset_abs.get_pct(Y, index)
    #     print(f"Index: {index}, Reversed pct: {reversed_pct}")