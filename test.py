'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-22 10:30:28
@LastEditors: GuoYi
'''
import torch
import os
import sys
import numpy as np
import time
from torch.optim import lr_scheduler
from torch import optim

from utils import check_dir
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from model import WGAN_SACNN_AE

from test_function import test_model
from exhibit_function import model_updata

class InitParser(object):
    def __init__(self):
        self.gpu_id = 3
        self.version = "v3"
        self.mode = "test"
        self.batch_size= {"train": 20, "val": 20, "test": 1}
        self.model_index = 275

        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 20

        self.re_load = False

        self.is_shuffle = True if self.mode is "train" else False
        self.data_length = {"train":5000, "val":500, "test":200}
        batch_num = {x:int(self.data_length[x]/self.batch_size[x]) for x in ["train", "val", "test"]}
        self.show_batch_num = {x:int(batch_num[x]/10) for x in ["train", "val", "test"]}

        # path setting
        if torch.cuda.is_available():
            self.data_root_path = "/mnt/tabgha/users/gy/data/Mayo/mayo_mat"
            self.root_path = "/mnt/tabgha/users/gy/MyProject/WGAN_SACNN_AE" 
        else:
            self.data_root_path = "V:/users/gy/data/Mayo/mayo_mat"
            self.root_path = "V:/users/gy/MyProject/WGAN_SACNN_AE"
        self.model_name = "WGAN_SACNN_AE_E"

        ## Calculate corresponding parameters
        self.result_path = self.root_path + "/results/{}".format(self.version)
        self.loss_path = self.result_path + "/loss"
        self.model_path = self.result_path + "/model"
        self.optimizer_path = self.result_path + "/optimizer"
        self.test_result_path = self.result_path + "/test_result"
        # self.train_folder = ["L192","L286","L291","L310","L333", "L506"]
        # self.test_folder = ["L067", "L096","L109","L143"]
        # self.val_folder = ["L067", "L096","L109","L143"]
        self.train_folder = ["L192"]
        self.test_folder = ["L067"]
        self.val_folder = ["L067"]

        if self.re_load or self.mode is "test":
            self.old_version = "v2"
            self.old_result_path = self.root_path + "/results/{}".format(self.old_version)
            self.old_modle_path = self.old_result_path + "/model"
            self.old_modle_name = self.model_name + str(299) + "_val_Best"



def main(args):
    if args.use_cuda:
        print("Using GPU")
        torch.cuda.set_device(args.gpu_id)
    else: 
        print("Using CPU")

    datasets = {"test": BuildDataSet(args.data_root_path, args.test_folder, None, args.data_length["test"], "test")}
    
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    dataloaders = {x: DataLoader(datasets[x], args.batch_size[x], shuffle=True, **kwargs) for x in ["test"]}
    print("Load Datasets Done")

    ## *********************************************************************************************************
    model_index = args.model_index
    model = WGAN_SACNN_AE(args.batch_size[args.mode], args.root_path, args.version)
    model = model_updata(model, model_old_name=args.model_name + "{}".format(model_index), model_old_path=args.model_path)

    ## *********************************************************************************************************
    test_model(model = model,
            dataloaders = dataloaders,
            args = args)

     
if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")



"""
Quarter:[8.99328911e-01 1.17713452e-03 5.71141977e+01]
Pred:[8.81595917e-01 2.16356885e-03 7.52886853e+01]

"""
