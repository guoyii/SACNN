'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-03 22:22:14
@LastEditors: GuoYi
'''
import numpy as np
import pydicom
import torch
import math 
import glob
import astra
from scipy.io import loadmat

"""
***********************************************************************************************************
Fixed image processing:
RandomCrop, Normalize, Any2One, ToTensor
***********************************************************************************************************
"""

## Cut image randomly
##***********************************************************************************************************
class RandomCrop(object):
    def __init__(self, crop_size, crop_point):
        self.crop_size = crop_size
        self.crop_point = crop_point

    def __call__(self, image):
        # image = np.hstack((image, image))
        # image = np.vstack((image, image))

        # crop_point = np.random.randint(self.crop_size, size=2)
        image = image[self.crop_point[0]:self.crop_point[0]+self.crop_size, self.crop_point[1]:self.crop_point[1]+self.crop_size]
        image = np.pad(image,((math.ceil((self.crop_size - image.shape[0])/2), math.floor((self.crop_size - image.shape[0])/2)),
                (math.ceil((self.crop_size - image.shape[1])/2), math.floor((self.crop_size - image.shape[1])/2))),"constant")
        return image


## Normalize
##***********************************************************************************************************
class Normalize(object):
    def __init__(self, normalize_type="image"):
        if normalize_type is "image":
            self.mean = 128.0
            # self.mean = 0.0 
        elif normalize_type is "self":
            self.mean = None

    def __call__(self, image):
        if self.mean is not None:
            img_mean = self.mean
        else:
            img_mean = np.mean(image)

        image = image - img_mean
        image = image / 255.0
        return image


class Scale2Gen(object):
    def __init__(self, scale_type="image"):
        if scale_type is "image":
            self.mmin = 0.0
            self.mmax = 2500.0

        elif scale_type is "self":
            self.mmin = None
            self.mmax = None

    def __call__(self, image):
        if self.mmin is not None:
            img_min, img_max = self.mmin, self.mmax
        else:
            img_min, img_max = np.min(image), np.max(image)
        
        image = (image - img_min) / (img_max-img_min) * 255.0
        return image


def normalize(image, img_mean=None, img_var=None):
    if img_mean is None or img_var is None:
        img_mean = torch.mean(image)
        img_var = torch.var(image)
    image = image - img_mean
    image = image / img_var

    return image, img_mean, img_var


## Image normalization
##***********************************************************************************************************
class Any2One(object):
    def __call__(self, image):
        image_max = torch.max(image)
        image_min = torch.min(image)
        return (image-image_min)/(image_max-image_min)


def any2one(image, image_max=None, image_min=None):
    if image_max is None or image_min is None:
        image_max = torch.max(image)
        image_min = torch.min(image)
    return (image-image_min)/(image_max-image_min), image_max, image_min


## Change to torch
##***********************************************************************************************************
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # return torch.from_numpy(image.astype(np.int16)).type(torch.FloatTensor)
        return torch.from_numpy(image).type(torch.FloatTensor)


"""
***********************************************************************************************************
Read Image:
read_mat
***********************************************************************************************************
"""
## Read Mat Files
##***********************************************************************************************************
def read_mat(path):
    image = loadmat(path)["Img_CT"]
    return image.transpose(2, 0, 1)



"""
***********************************************************************************************************
Training pretreatment:
Transpose, TensorFlip, MayoTrans
***********************************************************************************************************
"""
## Transpose
##***********************************************************************************************************
class Transpose(object):
    def __call__(self, image):
        return image.transpose_(1, 0)


## Flip
##***********************************************************************************************************
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ("cpu","cuda")[x.is_cuda])().long(), :]
    return x.view(xsize)


## Flip
##***********************************************************************************************************
class TensorFlip(object):
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, image):
        return flip(image, self.dim)


## 
##***********************************************************************************************************
class MayoTrans(object):
    def __init__(self, WaterAtValue, trans_style="self"):
        self.WaterAtValue = WaterAtValue
        self.AtValue2CTnum = AtValue2CTnum(WaterAtValue)
        self.Scale2Gen = Scale2Gen(trans_style)
        self.Normalize = Normalize(trans_style)

    def __call__(self, image):
        image = self.AtValue2CTnum(image)
        image, img_min, img_max = self.Scale2Gen(image)
        image, img_mean = self.Normalize(image)

        a = 1000.0/((img_max-img_min)*self.WaterAtValue)
        b = -(img_min+1000.0)/(img_max-img_min)-img_mean/255.0
 
        return image, a, b

        
"""
***********************************************************************************************************
Others:
Transpose, TensorFlip, MayoTrans
***********************************************************************************************************
"""
## Calculate the CT value from Calculate pixel value
##***********************************************************************************************************
class AtValue2CTnum(object):
    def __init__(self, WaterAtValue):
        self.WaterAtValue = WaterAtValue

    def __call__(self, image):
        image = (image -self.WaterAtValue)/self.WaterAtValue *1000.0
        return image


## Calculate the Calculate pixel value from CT value
##***********************************************************************************************************
class CTnum2AtValue(object):
    def __init__(self, WaterAtValue):
        self.WaterAtValue = WaterAtValue

    def __call__(self, image):
        image = image *self.WaterAtValue /1000.0 + self.WaterAtValue
        return image


## 
##***********************************************************************************************************
class SinoTrans(object):
    def __init__(self, trans_style="self"):
        self.Normalize = Normalize(trans_style)

    def __call__(self, sino):
        sino, img_mean = self.Normalize(sino)

        a = 1.0/255.0
        b = -img_mean/255.0

        return sino, a, b


## Add noise
##***********************************************************************************************************
def add_noise(noise_typ, image, mean=0, var=0.1):
    if noise_typ == "gauss":
        row,col= image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        noisy = np.random.poisson(image)
        return noisy

