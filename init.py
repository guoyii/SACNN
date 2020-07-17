'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-08 20:57:37
@LastEditors: GuoYi
'''
import torch 

class InitParser(object):
    def __init__(self):
        self.gpu_id = 2
        self.version = "v9"
        self.mode = "train"
        self.batch_size= {"train": 20, "val": 20, "test": 1}
        self.n_d_train = 4                                              ## Every training the generator, training the discriminator n times
        self.use_p_loss = False        
        self.re_load = False  

        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 20

        ## set optimizer
        self.lr = 0.00001
        self.momentum = 0.9
        self.weight_decay = 0.0
        
        ## set parameters
        self.epoch_num = 500

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
        self.optimizer_g_name = "OptimG_E"
        self.optimizer_d_name = "OptimD_E"

        ## Calculate corresponding parameters
        self.result_path = self.root_path + "/results/{}".format(self.version)
        self.loss_path = self.result_path + "/loss"
        self.model_path = self.result_path + "/model"
        self.optimizer_path = self.result_path + "/optimizer"
        self.test_result_path = self.result_path + "/test_result"
        self.train_folder = ["L192","L286","L291","L310","L333", "L506"]
        self.test_folder = ["L067", "L096","L109","L143"]
        self.val_folder = ["L067", "L096","L109","L143"]

        if self.re_load or self.mode is "test":
            self.old_version = "v1"
            self.old_index = 315
            self.old_result_path = self.root_path + "/results/{}".format(self.old_version)
            self.old_modle_path = self.old_result_path + "/model"
            self.old_optimizer_path = self.old_result_path + "/optimizer"
            self.old_modle_name = self.model_name + str(self.old_index)
            self.old_optimizer_g_name = self.optimizer_g_name + str(self.old_index)
            self.old_optimizer_d_name = self.optimizer_d_name + str(self.old_index)
