'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-13 09:45:03
@LastEditors: GuoYi
'''
import torch
import astra
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from datasets_function import RandomCrop, ToTensor, Normalize, Scale2Gen
from datasets_function import read_mat


## Basic datasets
##***********************************************************************************************************
class BasicData(Dataset):
    def __init__(self, data_root_path, folder, data_length, Dataset_name):
        self.folder = folder
        self.Dataset_name = Dataset_name
        
        self.data_length = data_length                                                                            ## The data size of each epoch
        self.Full_Image = {x:read_mat(data_root_path + "/{}_full_1mm_CT.mat".format(x)) for x in self.folder}        ## High-dose images of all patients
        self.Quarter_Image = {x:read_mat(data_root_path + "/{}_quarter_1mm_CT.mat".format(x)) for x in self.folder}  ## Low-dose images of all patients

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        patient_index = np.random.randint(len(self.folder))                     ## One image of the patient was chosen at a time
        full_image_set = self.Full_Image[self.folder[patient_index]]
        quarter_image_set = self.Quarter_Image[self.folder[patient_index]]
        
        image_index = np.random.randint(1, full_image_set.shape[0]-1)          ## Three consecutive images were randomly selected from people with the disease
        # print(image_index)
        if self.Dataset_name is "test":
            image_index = 66
        full_image = full_image_set[image_index-1:image_index+2]
        quarter_image = quarter_image_set[image_index-1:image_index+2]

        return full_image, quarter_image


class BuildDataSet(Dataset):
    def __init__(self, data_root_path, folder, pre_trans_img=None, data_length=None, Dataset_name="train", patch_size=64):
        self.Dataset_name = Dataset_name
        self.pre_trans_img = pre_trans_img
        self.imgset = BasicData(data_root_path, folder, data_length, Dataset_name=self.Dataset_name)
        self.patch_size = patch_size


    def __len__(self):
        return len(self.imgset)
    
    @classmethod
    def Cal_transform(cls, Dataset_name, pre_trans_img, fix_list):
        random_list = []
        if Dataset_name is "train":
            if pre_trans_img is not None:
                keys = np.random.randint(2,size=len(pre_trans_img))
                for i, key in enumerate(keys):
                    random_list.append(pre_trans_img[i]) if key == 1 else None
        transform = transforms.Compose(fix_list + random_list)
        return transform

    @classmethod
    def preProcess(cls, image, Transf, patch_size):
        new_image = torch.zeros((3, patch_size, patch_size))
        for i in range(3):  
            new_image[i] = Transf(image[i])
        return new_image
    
    def __getitem__(self, idx):
        full_image, quarter_image = self.imgset[idx]

        if self.patch_size != 64:
            crop_point = [0, 0]
        else:
            crop_point = np.random.randint(512, size=2)                            ## Random interception patch
            
        fix_list = [Scale2Gen(scale_type="image"), Normalize(normalize_type="image"), RandomCrop(self.patch_size, crop_point), ToTensor()]
        transf = self.Cal_transform(self.Dataset_name, self.pre_trans_img, fix_list)

        full_image = self.preProcess(full_image, transf, patch_size=self.patch_size)
        quarter_image = self.preProcess(quarter_image, transf, patch_size=self.patch_size)
        
        sample = {"full_image": full_image.unsqueeze_(0), 
                "quarter_image": quarter_image.unsqueeze_(0)}
        return sample

        