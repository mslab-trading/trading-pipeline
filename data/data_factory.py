from .data_loader import Dataset_Abs

from torch.utils.data import DataLoader
import json
import argparse

def data_provider(args, flag):
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
        
    stock_ids = basic_info[args.category]
    
    if args.data == 'Dataset_Abs':
        data_set = Dataset_Abs(
            root_dir_path=args.root_path,
            broker_dir_path = args.broker_path,
            general_data_path = args.general_data_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=args.split_dates,
            goal=args.goal,
            log=args.log,
            thresh=args.thresh,
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
    
