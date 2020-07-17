'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-24 14:57:23
@LastEditors: GuoYi
'''
import torch
import os
import sys
import numpy as np
import time
from torch.optim import lr_scheduler
from torch import optim

from init import InitParser
from utils import check_dir, updata_model
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from model import WGAN_SACNN_AE
from train_function import train_model


def main(args):
    if args.use_cuda:
        print("Using GPU")
        torch.cuda.set_device(args.gpu_id)
    else: 
        print("Using CPU")

    check_dir(args.loss_path)
    check_dir(args.model_path)
    check_dir(args.optimizer_path)

    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    datasets = {"train": BuildDataSet(args.data_root_path, args.train_folder, pre_trans_img, args.data_length["train"], "train"),
            "val": BuildDataSet(args.data_root_path, args.val_folder, None, args.data_length["val"], "val")}

    data_length = {x:len(datasets[x]) for x in ["train", "val"]}
    print("Data length:Train:{} Val:{}".format(data_length["train"], data_length["val"]))
    
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    dataloaders = {x: DataLoader(datasets[x], args.batch_size[x], shuffle=args.is_shuffle, **kwargs) for x in ["train", "val"]}

    ## *********************************************************************************************************
    model = WGAN_SACNN_AE(args.batch_size[args.mode], args.root_path, "v2")
    if args.mode is "train":
        train_model(model = model,
                dataloaders = dataloaders,
                args=args
                )
        print("Run train.py Success!")
    else:
        print("\nargs.mode is wrong!\n")
        sys.exit(0)
    
if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done\n")
