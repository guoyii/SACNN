'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-22 10:32:06
@LastEditors: GuoYi
'''
import os
import time
import math
import pickle
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
from torch.autograd import Variable
from exhibit_function import ssim_mse_psnr
from utils import check_dir

def test_model(model,
              dataloaders,
              args):
        check_dir(args.test_result_path)

        batch_num = int(args.data_length["test"]/args.batch_size["test"])
        result = np.zeros((batch_num+1,6))
        print("Result Shape:{}".format(result.shape))
        ## quarter_ssim,quarter_mse,quarter_psnr,quarter_loss,pred_ssim,pred_mse,pred_psnr,pred_loss
        ## The last is average value
        quarter_ssim_avg = 0
        quarter_mse_avg = 0
        quarter_psnr_avg = 0

        pred_ssim_avg = 0
        pred_mse_avg = 0
        pred_psnr_avg = 0

        time_all_start = time.time()
        model.eval()
        print("**************  Test  ****************")
        for i, batch in enumerate(dataloaders["test"]):
                # print("Now testing {} sample......".format(i))
                quarter_image = batch["quarter_image"]
                full_image = batch["full_image"]

                if args.use_cuda:
                        full_image = Variable(full_image).cuda()
                        quarter_image = Variable(quarter_image).cuda()
                else:
                        full_image = Variable(full_image)
                        quarter_image = Variable(quarter_image)

                with torch.no_grad():
                        image_pred = model.generator(quarter_image)
                
                s_ssim,s_mse,s_psnr = ssim_mse_psnr(quarter_image.cpu().numpy()[0,0,1,:,:], full_image.cpu().numpy()[0,0,1,:,:])
                p_ssim,p_mse,p_psnr = ssim_mse_psnr(image_pred.cpu().numpy()[0,0,1,:,:], full_image.cpu().numpy()[0,0,1,:,:])
                
                quarter_ssim_avg += s_ssim
                quarter_mse_avg += s_mse
                if s_psnr<200:
                        quarter_psnr_avg += s_psnr
                else:
                        quarter_psnr_avg += 0

                pred_ssim_avg += p_ssim
                pred_mse_avg += p_mse
                if p_psnr<200:
                        pred_psnr_avg += p_psnr
                else:
                        pred_psnr_avg += 0
        
                result[i] = [s_ssim,s_mse,s_psnr,p_ssim,p_mse,p_psnr]
                if math.fmod(i, 100) == 0:
                        print("Now testing {}-{} sample......".format(i, i+100))

        quarter_ssim_avg = quarter_ssim_avg/args.data_length["test"]
        quarter_mse_avg = quarter_mse_avg/args.data_length["test"]
        quarter_psnr_avg = quarter_psnr_avg/args.data_length["test"]

        pred_ssim_avg = pred_ssim_avg/args.data_length["test"]
        pred_mse_avg = pred_mse_avg/args.data_length["test"]
        pred_psnr_avg = pred_psnr_avg/args.data_length["test"]
        
        result[-1] = [quarter_ssim_avg,quarter_mse_avg,quarter_psnr_avg,
                        pred_ssim_avg,pred_mse_avg,pred_psnr_avg]
        """
        np.save("filename.npy",a)
        b = np.load("filename.npy")
        """
        np.save(args.test_result_path + "/results.npy", result)
        print("SSIM   MSE   PSNR")
        print("Quarter:{}".format(result[-1][0:3]))
        print("Pred:{}".format(result[-1][3:6]))
        print("Test completed ! Time is {:.4f}min".format((time.time() - time_all_start)/60)) 
