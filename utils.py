'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-17 15:39:21
@LastEditors: GuoYi
'''
import os
import torch
import time 
import numpy as np 
import torch.distributed as dist


## Check the path
##***********************************************************************************************************
def check_dir(path):

	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)


## Many GPU training
##***********************************************************************************************************
def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"   # 断言函数 raise if not
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus

    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
        print("ngpus:",ngpus)
    elif ngpus == 1:
        model = model.cuda()
    return model


## Updata old model
##***********************************************************************************************************
def updata_model(model, args):
     print("Please set the path of expected model!")
     time.sleep(3)
     old_modle_path = args.old_modle_path
     model_reload_path = old_modle_path + "/" + args.old_modle_name + ".pkl"

     if os.path.isfile(model_reload_path):
          print("Loading previously trained network...")
          print("Load model:{}".format(model_reload_path))
          checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
          model_dict = model.state_dict()
          checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
          model_dict.update(checkpoint)
          model.load_state_dict(model_dict)
          del checkpoint
          torch.cuda.empty_cache()
          if args.use_cuda:
               model = model.cuda()
          print("Done Reload!")
     else:
          print("Can not reload model..../n")
          time.sleep(10)
          sys.exit(0)
     return model


## Updata ae
##***********************************************************************************************************
def updata_ae(model, ae_path):
     print("\n......Please set the path of AE!......")
     if os.path.isfile(ae_path):
          print("Ae set done, Loading...")
          print("Load model:{}".format(ae_path))
          checkpoint = torch.load(ae_path, map_location = lambda storage, loc: storage)
          model_dict = model.state_dict()
          checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
          model_dict.update(checkpoint)
          model.load_state_dict(model_dict)
          del checkpoint
          torch.cuda.empty_cache()
          if torch.cuda.is_available():
               model = model.cuda()
          print("Ae Reload!\n")
     else:
          print("Can not reload Ae....\n")
          time.sleep(10)
          sys.exit(0)
     return model

