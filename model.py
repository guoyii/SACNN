'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-08 10:24:49
@LastEditors: GuoYi
'''
import torch.nn.functional as F
from model_function import SA, Conv_3d, OutConv
from model_function import AE_Down, AE_Up
from torch import nn
from utils import updata_ae
import torch 
import numpy as np 
import time 

"""
Generator
"""
##******************************************************************************************************************************
class SACNN(nn.Module):
    def __init__(self, N, version):
        super(SACNN, self).__init__()
        self.input_channels = 1
        self.output_channels = 1
        self.N = N
        
        if version is "v2": 
            self.lay1 = Conv_3d(in_ch=self.input_channels, out_ch=64, use_relu="use_relu")
            self.lay2 = Conv_3d(in_ch=64, out_ch=32, use_relu="use_relu")
            # self.lay3 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay3 = Conv_3d(in_ch=32, out_ch=32, use_relu="use_relu")
            self.lay4 = Conv_3d(in_ch=32, out_ch=16, use_relu="use_relu")
            # self.lay5 = SA(in_ch=2, out_ch=2, N=self.N)
            self.lay5 = Conv_3d(in_ch=16, out_ch=16, use_relu="use_relu")
            self.lay6 = Conv_3d(in_ch=16, out_ch=32, use_relu="use_relu")
            # self.lay7 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay7 = Conv_3d(in_ch=32, out_ch=32, use_relu="use_relu")
            self.lay8 = Conv_3d(in_ch=32, out_ch=64, use_relu="use_relu")
            self.lay9 = OutConv(in_ch=64, out_ch=self.output_channels)
        elif version is "v1":
            self.lay1 = Conv_3d(in_ch=self.input_channels, out_ch=8, use_relu="use_relu")
            self.lay2 = Conv_3d(in_ch=8, out_ch=4, use_relu="use_relu")
            # self.lay3 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay3 = Conv_3d(in_ch=4, out_ch=4, use_relu="use_relu")
            self.lay4 = Conv_3d(in_ch=4, out_ch=2, use_relu="use_relu")
            # self.lay5 = SA(in_ch=2, out_ch=2, N=self.N)
            self.lay5 = Conv_3d(in_ch=2, out_ch=2, use_relu="use_relu")
            self.lay6 = Conv_3d(in_ch=2, out_ch=4, use_relu="use_relu")
            # self.lay7 = SA(in_ch=4, out_ch=4, N=self.N)
            self.lay7 = Conv_3d(in_ch=4, out_ch=4, use_relu="use_relu")
            self.lay8 = Conv_3d(in_ch=4, out_ch=8, use_relu="use_relu")
            self.lay9 = OutConv(in_ch=8, out_ch=self.output_channels)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)
        x = self.lay8(x)
        x = self.lay9(x)
        return x


"""
Perceptual loss
"""
##******************************************************************************************************************************
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.lay1 = AE_Down(in_channels=1, out_channels=64)
        self.lay2 = AE_Down(in_channels=64, out_channels=128)
        self.lay3 = AE_Down(in_channels=128, out_channels=256)
        self.lay4 = AE_Down(in_channels=256, out_channels=256)

        self.lay5 = AE_Up(in_channels=256, out_channels=256)
        self.lay6 = AE_Up(in_channels=256, out_channels=128)
        self.lay7 = AE_Up(in_channels=128, out_channels=64)
        self.lay8 = AE_Up(in_channels=64, out_channels=32)
        self.lay9 = OutConv(in_ch=32, out_ch=1)

        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.deconv1 = nn.ConvTranspose3d(128, 128, kernel_size=(1,2,2), stride=(1,2,2))
        self.deconv2 = nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2))
    
    def forward(self, x):
        x = self.lay1(x)
        x = self.maxpool(x)
        x = self.lay2(x)
        x = self.maxpool(x)
        x = self.lay3(x)
        y = self.lay4(x)

        x = self.lay5(y)
        x = self.lay6(x)
        x = self.deconv1(x)
        x = self.lay7(x)
        x = self.deconv2(x)
        x = self.lay8(x)
        out = self.lay9(x)
        return out, y


"""
Discriminator
"""
##******************************************************************************************************************************
class DISC(nn.Module):
    def __init__(self):
        super(DISC, self).__init__()

        self.lay1 = Conv_3d(in_ch=1, out_ch=16, use_relu="no")
        self.lay2 = Conv_3d(in_ch=16, out_ch=32, use_relu="no")
        self.lay3 = Conv_3d(in_ch=32, out_ch=64, use_relu="no")
        self.lay4 = Conv_3d(in_ch=64, out_ch=64, use_relu="no")
        self.lay5 = Conv_3d(in_ch=64, out_ch=32, use_relu="no")
        self.lay6 = Conv_3d(in_ch=32, out_ch=16, use_relu="no")
        self.lay7 = Conv_3d(in_ch=16, out_ch=1, use_relu="no")

        ## out.view(-1, 256*self.output_size*self.output_size)
        self.fc1 = nn.Linear(3*64*64, 1024)    ## input:N*C*D*H*W=N*1*3*64*64 
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)
        x = self.lay7(x)

        x = self.fc1(x.view(-1, 3*64*64))
        x = self.fc2(x)
        return x


"""
Whole Network
"""
##******************************************************************************************************************************
class WGAN_SACNN_AE(nn.Module):
    def __init__(self, N, root_path, version="v2"):
        super(WGAN_SACNN_AE, self).__init__()
        if torch.cuda.is_available():
            self.generator = SACNN(N, version).cuda()
            self.discriminator = DISC().cuda()
            self.p_criterion = nn.MSELoss().cuda()
        else:
            self.generator = SACNN(N, version)
            self.discriminator = DISC()
            self.p_criterion = nn.MSELoss()
        ae_path = root_path + "/AE/Ae_E279_val_Best.pkl"             ## The network has been trained to compute perceputal loss
        Ae = AE()
        self.ae = updata_ae(Ae, ae_path)

    def feature_extractor(self, image, model):
        model.eval()
        pred,y = model(image)
        return y

    def d_loss(self, x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)

        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        """
        generator loss
        """
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        mse_loss = self.p_criterion(x, y)
        g_loss += mse_loss*100
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        """
        percetual loss
        """
        fake = self.generator(x)
        real = y
        fake_feature = self.feature_extractor(fake, self.ae)
        real_feature = self.feature_extractor(real, self.ae)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()

        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

