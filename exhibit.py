'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-13 10:24:00
@LastEditors: GuoYi
'''
import torch 
import pickle
import copy
import numpy as np 
import matplotlib.pylab as plt 
from torch.utils.data import DataLoader 
from scipy.io import loadmat
import scipy.io as scio

from datasets_function import Transpose, TensorFlip, MayoTrans
# from datasets_function import normalize, any2one
from datasets import BuildDataSet

from exhibit_function import ssim_mse_psnr, rec_image
from exhibit_function import read_loss, show_loss, show_Wasserstein
from exhibit_function import model_updata, pred_sample
from model import WGAN_SACNN_AE

class InitParser(object):
    def __init__(self):
        self.version = "v7"
        self.model_index = 420

        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 20
        self.model_version = "v2"

        self.re_load = False

        self.gpu_id = 3
        self.mode = "test"
        self.batch_size= {"train": 20, "val": 20, "test": 1}

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
        self.test_folder = ["L192"]
        self.val_folder = ["L192"]


def main(args):
    print("-"*15, "Version:{}".format(args.version), "-"*15)
    print("*"*50)

    index = np.random.randint(low=0, high=args.data_length["test"])
    datasets = {"test": BuildDataSet(args.data_root_path, args.test_folder, None, args.data_length["test"], "test", patch_size=512)}

    sample = datasets["test"][index]
    full_image = sample["full_image"]
    quarter_image = sample["quarter_image"]
    quarter_pred_image = copy.copy(quarter_image)
    
    """
    ***********************************************************************************************************
    Show Loss
    ***********************************************************************************************************
    """
    loss_train, loss_val = read_loss("losses", args.loss_path)
    Wasserstein = -(loss_train[:,2] + loss_train[:, 3])
    show_Wasserstein(Wasserstein)
    show_loss(loss_train)
    plt.show()


    """
    ***********************************************************************************************************
    Test model
    ***********************************************************************************************************
    # """ 
    model_index = args.model_index
    model = WGAN_SACNN_AE(args.batch_size[args.mode], args.root_path, args.model_version)
    model = model_updata(model, model_old_name=args.model_name + "{}".format(model_index), model_old_path=args.model_path)
    pred_image = pred_sample(quarter_pred_image, model.generator)
    print("Full_Image Shape", full_image.shape)
    print("Quarter_Image Shape", quarter_image.shape)
    print("Pred_Image Shape", pred_image.shape)
    if 1 :
        full_image = rec_image(full_image[0,:,:,:])
        quarter_image = rec_image(quarter_image[0,:,:,:])
        pred_image = rec_image(pred_image[0,:,:,:])
    else:
        full_image = full_image[0,:,:,:].numpy()
        quarter_image = quarter_image[0,:,:,:].numpy()
        pred_image = pred_image[0,:,:,:].numpy()

    scio.savemat(args.result_path + "/image.mat", {'full_image':full_image, 'quarter_image':quarter_image, 'pred_image':pred_image})

    """
    ***********************************************************************************************************
    量化指标
    ***********************************************************************************************************
    """ 
    # ssim_0, mse_0, psnr_0 = ssim_mse_psnr(full_image[1,:,:], quarter_image[1,:,:])
    # ssim_1, mse_1, psnr_1 = ssim_mse_psnr(full_image[1,:,:], pred_image[1,:,:])
    # print("Quarter Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_0, mse_0, psnr_0))
    # print("Predict Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_1, mse_1, psnr_1))

    
    # """
    # ***********************************************************************************************************
    # Show images
    # ***********************************************************************************************************
    # """
    # plt.figure()
    # plt.subplot(331), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[0,:,:], cmap="gray"),                          plt.title("full_image")
    # plt.subplot(332), plt.xticks([]), plt.yticks([]), plt.imshow(quarter_image[0,:,:], cmap="gray"),                       plt.title("quarter_image")
    # plt.subplot(333), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[0,:,:]-quarter_image[0,:,:], cmap="gray"),   plt.title("res_image")
    # plt.subplot(334), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[1,:,:], cmap="gray"),                          plt.title("full_image")
    # plt.subplot(335), plt.xticks([]), plt.yticks([]), plt.imshow(quarter_image[1,:,:], cmap="gray"),                       plt.title("quarter_image")
    # plt.subplot(336), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[1,:,:]-quarter_image[1,:,:], cmap="gray"),   plt.title("res_image")
    # plt.subplot(337), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[2,:,:], cmap="gray"),                          plt.title("full_image")
    # plt.subplot(338), plt.xticks([]), plt.yticks([]), plt.imshow(quarter_image[2,:,:], cmap="gray"),                       plt.title("quarter_image")
    # plt.subplot(339), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[2,:,:]-quarter_image[2,:,:], cmap="gray"),   plt.title("res_image")
    # plt.show()

    plt.figure()
    plt.subplot(231), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[1,:,:], cmap="gray"),                          plt.title("Full image")
    plt.subplot(232), plt.xticks([]), plt.yticks([]), plt.imshow(quarter_image[1,:,:], cmap="gray"),                       plt.title("Quarter image")
    plt.subplot(233), plt.xticks([]), plt.yticks([]), plt.imshow(pred_image[1,:,:], cmap="gray"),                          plt.title("Pred image")
    plt.subplot(234), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[1,:,:]-full_image[1,:,:], cmap="gray"),      plt.title("Full res image")
    plt.subplot(235), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[1,:,:]-quarter_image[1,:,:], cmap="gray"),   plt.title("Quarter res image")
    plt.subplot(236), plt.xticks([]), plt.yticks([]), plt.imshow(full_image[1,:,:]-pred_image[1,:,:], cmap="gray"),      plt.title("Pred res image")
    plt.show()


if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")