from .data_loader import Dataset_Abs, Dataset_Pct, Dataset_S3E, Dataset_Jerome, Dataset_Berlin
from .data_reader import read_market_data, read_broker_data, read_global_data

from torch.utils.data import DataLoader
import json
import argparse

def data_provider(args, flag, isS3E=False):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
    
    with open(args.data_info_path, "r") as f:
        basic_info = json.load(f)
    
    with open(args.broker_info_path, "r") as f:
        broker_info = json.load(f)
    
    broker_names = broker_info['rank'][:args.broker_topK]
           
    stock_ids = basic_info[args.category]
    market_features = args.market_features
    global_features = args.global_features
    
    market_df = read_market_data(
        args.root_path,
        stock_ids,
        global_data_path=args.general_data_path if args.concat_market_global else None,
        market_features=market_features,
        global_features=global_features
    )

    global_df = read_global_data(args.general_data_path, global_features=global_features)
    broker_df = read_broker_data(args.broker_path, stock_ids, broker_names)

    if isS3E:
        data_set = Dataset_S3E(
            data=args.data,
            market_df=market_df,
            broker_df=broker_df,
            global_df=global_df,
            size=[args.seq_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=args.split_dates,
            goal=args.goal,
            log=args.log,
            thresh=args.thresh,
        )
    elif args.data == 'Dataset_Abs':
        data_set = Dataset_Abs(
            market_df=market_df,
            broker_df=broker_df,
            global_df=global_df,
            size=[args.seq_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=args.split_dates,
            goal=args.goal,
            log=args.log,
            thresh=args.thresh,
        )
    elif args.data == 'Dataset_Pct':
        data_set = Dataset_Pct(
            market_df=market_df,
            broker_df=broker_df,
            global_df=global_df,
            size=[args.seq_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=args.split_dates,
            goal=args.goal,
            log=args.log,
            thresh=args.thresh,
        )
    
    # TODO: jerome
    elif args.data == 'Dataset_Jerome':
        data_set = Dataset_Jerome(
        )
    
    # TODO: berlin
    elif args.data == 'Dataset_Berlin':
        data_set = Dataset_Berlin(
        )
        
    else:
        raise ValueError(f"Invalid data type: {args.data}")
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader
    
