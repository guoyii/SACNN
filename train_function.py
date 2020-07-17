'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-28 20:50:56
@LastEditors: GuoYi
'''
import os
import sys
import time
import math
import pickle
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
from torch import optim
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

## train function
##***********************************************************************************************************
def train(model, epoch, phase, optimizer_g, optimizer_d, dataloaders, args):
     g_loss_all = 0
     p_loss_all = 0
     d_loss_all = 0
     gp_loss_all = 0
     for i, batch in enumerate(dataloaders):
          time_batch_start = time.time()
          full_image = batch["full_image"]
          quarter_image = batch["quarter_image"]

          if args.use_cuda:
               full_image = Variable(full_image).cuda()
               quarter_image = Variable(quarter_image).cuda()
          else:
               full_image = Variable(full_image)
               quarter_image = Variable(quarter_image)
          
          ##**********************************
          # (1)discriminator
          ##**********************************
          optimizer_d.zero_grad()
          model.discriminator.zero_grad()

          for _ in range(args.n_d_train):
               loss1, gp_loss = model.d_loss(quarter_image, full_image, gp=True, return_gp=True)
               loss1.backward()
               optimizer_d.step()

               d_loss_all += ((loss1-gp_loss)*quarter_image.size(0) /args.n_d_train)  ## loss = d_loss + gp_loss
               gp_loss_all += (gp_loss*quarter_image.size(0) /args.n_d_train)

          
          ##**********************************
          # (2)generator
          ##**********************************
          optimizer_g.zero_grad()
          model.generator.zero_grad()

          loss2, p_loss = model.g_loss(quarter_image, full_image, perceptual=args.use_p_loss, return_p=True)
          loss2.backward()
          optimizer_g.step()

          if args.use_p_loss is False:
               p_loss = 0
          g_loss_all += (loss2 - (0.1 * p_loss))*quarter_image.size(0)   ## loss = g_loss + (0.1 * p_loss)
          p_loss_all += p_loss*quarter_image.size(0)       


          if i>0 and math.fmod(i, args.show_batch_num[phase]) == 0:
               print("Epoch {} Batch {}-{} {}, Time:{:.4f}s".format(epoch+1,
               i-args.show_batch_num[phase], i, phase, (time.time()-time_batch_start)*args.show_batch_num[phase]))

     g_loss = g_loss_all/args.data_length[phase]
     p_loss = p_loss_all/args.data_length[phase]
     d_loss = d_loss_all/args.data_length[phase]
     gp_loss = gp_loss_all/args.data_length[phase]
     return g_loss,p_loss,d_loss,gp_loss


## Train
##***********************************************************************************************************
def train_model(model,
                dataloaders,
                args): 

     optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
     optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

     # optimizer_g = optim.RMSprop(model.generator.parameters(), lr=args.lr, alpha=0.9)
     # optimizer_d = optim.RMSprop(model.discriminator.parameters(), lr=args.lr, alpha=0.9)

     if args.re_load is False:
          print("\nInit Start**")
          model.apply(weights_init)
          print("******Init End******\n")
     else:
          print("Re_load is True !")
          model, optimizer_g, optimizer_d = updata_model(model, optimizer_g, optimizer_d, args)
          
     losses = {x: torch.zeros(args.epoch_num, 4) for x in ["train", "val"]}
     # if args.re_load is True:
     #      losses = {x: torch.from_numpy(loadmat(args.loss_path + "/losses.mat")[x]) for x in ["train", "val"]}
     #      print("Load losses Done")
     

     temp = 0
     ##********************************************************************************************************************
     time_all_start = time.time()
     # for epoch in range(args.old_index if args.re_load else 0, args.epoch_num):
     for epoch in range(args.epoch_num):
          time_epoch_start = time.time()
          print("-" * 60)
          print(".........Training and Val epoch {}, all {} epochs..........".format(epoch+1, args.epoch_num))
          print("-" * 60)
          
          ##//////////////////////////////////////////////////////////////////////////////////////////////
          # for phase in ["train", "val"]:
          for phase in ["train"]:
               print("\n=========== Now, Start {}===========".format(phase))
               if phase is "train":
                    model.train()
               elif phase is "val":
                    model.eval()

               g_loss,p_loss,d_loss,gp_loss = train(model, epoch, phase, optimizer_g, optimizer_d, dataloaders[phase], args)
               losses[phase][epoch] = torch.tensor([g_loss,p_loss,d_loss,gp_loss])
    
          ##//////////////////////////////////////////////////////////////////////////////////////////////
          if math.fmod(epoch, 5) == 0:
               torch.save(model.state_dict(), args.model_path + "/" + args.model_name + "{}.pkl".format(epoch))
               torch.save(optimizer_g.state_dict(), args.optimizer_path + "/" + args.optimizer_g_name + "{}.pkl".format(epoch))
               torch.save(optimizer_d.state_dict(), args.optimizer_path + "/" + args.optimizer_d_name + "{}.pkl".format(epoch))

          data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in losses.items()}
          sio.savemat(args.loss_path + "/losses.mat", mdict = data_save)
    
          print("Time for epoch {} : {:.4f}min".format(epoch+1, (time.time()-time_epoch_start)/60))
          print("Time for ALL : {:.4f}h\n".format((time.time()-time_all_start)/3600))
     ##********************************************************************************************************************
     print("\nTrain Completed!! Time for ALL : {:.4f}h".format((time.time()-time_all_start)/3600))


## Init the model
##***********************************************************************************************************
def weights_init(m):
     classname = m.__class__.__name__
     if classname.find("Conv3d") != -1:
          init.xavier_normal_(m.weight.data)
          if m.bias is not None:
               init.constant_(m.bias.data, 0)
          print("Init {} Parameters.................".format(classname))
     if classname.find("Linear") != -1:
          init.xavier_normal(m.weight)
          print("Init {} Parameters.................".format(classname))
     else:
          print("{} Parameters Do Not Need Init !!".format(classname))


## Updata old model
##***********************************************************************************************************
def updata_model(model, optimizer_g, optimizer_d, args):
     print("Please set the path of expected model!")
     time.sleep(3)
     model_reload_path = args.old_modle_path + "/" + args.old_modle_name + ".pkl"
     optimizer_g_reload_path = args.old_optimizer_path + "/" + args.old_optimizer_g_name + ".pkl"
     optimizer_d_reload_path = args.old_optimizer_path + "/" + args.old_optimizer_d_name + ".pkl"

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
          print("Can not reload model....\n")
          time.sleep(10)
          sys.exit(0)
     
     if os.path.isfile(optimizer_g_reload_path):
          print("Loading previous optimizer...")
          print("Load optimizer:{}".format(optimizer_g_reload_path))
          checkpoint = torch.load(optimizer_g_reload_path, map_location = lambda storage, loc: storage)
          optimizer_g.load_state_dict(checkpoint)
          del checkpoint
          torch.cuda.empty_cache()
          print("Done Reload!")
     else:
          print("Can not reload optimizer_g....\n")
          time.sleep(10)
          sys.exit(0)

     if os.path.isfile(optimizer_d_reload_path):
          print("Loading previous optimizer...")
          print("Load optimizer:{}".format(optimizer_d_reload_path))
          checkpoint = torch.load(optimizer_d_reload_path, map_location = lambda storage, loc: storage)
          optimizer_d.load_state_dict(checkpoint)
          del checkpoint
          torch.cuda.empty_cache()
          print("Done Reload!")
     else:
          print("Can not reload optimizer_d....\n")
          time.sleep(10)
          sys.exit(0)

     return model, optimizer_g, optimizer_d
