'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-24 21:27:44
@LastEditors: GuoYi
'''
import numpy as np 
import torch 
import os
import sys
import matplotlib.pylab as plt
from scipy.io import loadmat 

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from datasets_function import Any2One 


## Recover
def rec_image(image):
    image = image.numpy()
    for i in range(3):
        img = image[i,:,:]
        img = img * 255.0
        img = img + 128.0
        img = (img / 255.0) * (2500.0 - 0.0) + 0.0
        image[i,:,:] = img
    return image


## Pred one sample
##******************************************************************************************************************************
def pred_sample(image_sparse, model):
    model.eval()
    with torch.no_grad():
        image_pred = model(image_sparse.unsqueeze_(0))
    return image_pred[0]


## Load model
##******************************************************************************************************************************
def model_updata(model, model_old_name, model_old_path):
    model_reload_path = model_old_path + "/" + model_old_name + ".pkl"
    print("\nWill loading trained {}, please check out the path!".format(model_old_name))
    if os.path.isfile(model_reload_path):
        print("Path is right, Loading...")
        checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
        model_dict = model.state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print("{} Load Done!\n".format(model_old_name))
        return model
    else:
        print("\nLoading Fail!\n")
        sys.exit(0)

## show WAss
##******************************************************************************************************************************
def show_Wasserstein(Wasserstein):
    limit = np.where(Wasserstein==0)[0][0]-1 if np.min(Wasserstein) == 0 else 500
    plt.figure()
    plt.plot(np.arange(Wasserstein.shape[0]), Wasserstein, color = "r", label="Wasserstein")
    plt.xlim(left=0, right=limit)
    plt.ylim(0)
    plt.xlabel("Epoch")
    plt.ylabel("Wasserstein-Distance")
    plt.title("Wasserstein")
    plt.show()

        
## show or return loss
##******************************************************************************************************************************
def show_loss(loss, show_all=True):
    g_loss = loss[:,0]
    p_loss = loss[:,1]
    d_loss = loss[:,2]
    gp_loss = loss[:,3]

    limit = np.where(gp_loss==0)[0][0]-1 if np.min(gp_loss) == 0 else 500
    # plt.xlim([0, limit])
    # plt.ylim(bottom=0)
    # # plt.ylim(bottom=0, top=0.01)
    # if show_all:
    #     plt.figure()
    #     plt.plot(np.arange(loss.shape[0]), abs(g_loss), color = "b", label="g_loss")
    #     plt.plot(np.arange(loss.shape[0]), p_loss, color = "c", label="p_loss")
    #     plt.plot(np.arange(loss.shape[0]), abs(d_loss), color = "y", label="d_loss")
    #     plt.plot(np.arange(loss.shape[0]), gp_loss, color = "r", label="gp_loss")
    #     plt.xlim(left=0, right=limit)
    #     plt.ylim(bottom=0)
    #     plt.legend()
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     # plt.show()
    # else:
    plt.figure()
    plt.subplot(221), plt.plot(np.arange(loss.shape[0]), g_loss, color = "r", label="g_loss"),  plt.xlim(left=0, right=limit), plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("G_Loss")
    plt.subplot(222), plt.plot(np.arange(loss.shape[0]), p_loss, color = "r", label="ploss"),   plt.xlim(left=0, right=limit), plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("P_Loss")
    plt.subplot(223), plt.plot(np.arange(loss.shape[0]), d_loss, color = "r", label="d_loss"),  plt.xlim(left=0, right=limit), plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("D_Loss")
    plt.subplot(224), plt.plot(np.arange(loss.shape[0]), gp_loss, color = "r", label="gp_loss"),plt.xlim(left=0, right=limit), plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("GP_Loss")

    plt.figure()
    plt.subplot(121), plt.plot(np.arange(loss.shape[0]), g_loss+0.1*p_loss, color = "r"),  plt.xlim(left=0, right=limit), plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Generator Loss")
    plt.subplot(122), plt.plot(np.arange(loss.shape[0]), d_loss+gp_loss, color = "r"),   plt.xlim(left=0, right=limit), plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Discriminator Loss")
    # plt.show()


## show ssim mse psnr
##******************************************************************************************************************************
def ssim_mse_psnr(image_true, image_test):
    mse = compare_mse(image_true, image_test)
    ssim = compare_ssim(image_true, image_test)
    psnr = compare_psnr(image_true, image_test, data_range=255)
    return ssim, mse, psnr
 

## read loss
##******************************************************************************************************************************
def read_loss(loss_name, loss_path):
    loss_data_path = loss_path + "/{}.mat".format(loss_name)
    losses = loadmat(loss_data_path)
    loss_train = losses["train"]
    loss_val = losses["val"]
    return loss_train, loss_val


## Image normalization
##******************************************************************************************************************************
def any2one(image):
    process = Any2One()
    image = image.numpy()
    for i in range(3):
        image[0,i,:,:] = process(image[0,i,:,:])
    return torch.from_numpy(image)

