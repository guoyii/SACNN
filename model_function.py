'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-28 17:43:02
@LastEditors: GuoYi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

## Self-Attention Block
##***********************************************************************************************************
class SA(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_ch, out_ch, N):
        super().__init__()
        self.N = N 
        self.C = in_ch
        self.D = 3
        self.H = 64
        self.W = 64
        self.gama = nn.Parameter(torch.tensor([0.0]))

        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.conv3d_3 = nn.Sequential(
            # Conv3d input:N*C*D*H*W
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3d_1 = nn.Sequential(
            # Conv3d input:N*C*D*H*W
            # Conv3d output:N*C*D*H*W
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(self.out_ch),
            nn.ReLU(inplace=True), 
        )


    @classmethod
    def Cal_Patt(cls, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        k_x_flatten = k_x.reshape((N, C, D, 1, H * W))
        q_x_flatten = q_x.reshape((N, C, D, 1, H * W))
        v_x_flatten = v_x.reshape((N, C, D, 1, H * W))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 4, 3), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=4)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Patt = torch.matmul(v_x_flatten, r_x).reshape(N, C, D, H, W)
        return Patt

    
    @classmethod
    def Cal_Datt(cls, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        # k_x_transpose = k_x.permute(0, 1, 3, 4, 2)
        # q_x_transpose = q_x.permute(0, 1, 3, 4, 2)
        # v_x_transpose = v_x.permute(0, 1, 3, 4, 2)
        k_x_flatten = k_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        q_x_flatten = q_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        v_x_flatten = v_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 3, 5, 4), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=5)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Datt = torch.matmul(v_x_flatten, r_x).reshape(N, C, H, W, D)
        return Datt.permute(0, 1, 4, 2, 3)

    
    def forward(self, x):
        v_x = self.conv3d_3(x)
        k_x = self.conv3d_1(x)
        q_x = self.conv3d_1(x)
        
        Patt = self.Cal_Patt(k_x, q_x, v_x, self.N, self.C, self.D, self.H, self.W)
        Datt = self.Cal_Datt(k_x, q_x, v_x, self.N, self.C, self.D, self.H, self.W)
        
        Y = self.gama*(Patt + Datt) + x
        return Y


## 3D Convolutional
##***********************************************************************************************************
class Conv_3d(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_ch, out_ch, use_relu="use_relu"):
        super().__init__()
        if use_relu is "use_relu":
            self.conv3d = nn.Sequential(
                # Conv3d input:N*C*D*H*W
                # Conv3d output:N*C*D*H*W
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv3d = nn.Sequential(
                nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.conv3d(x)
        return out


## Out Conv
##***********************************************************************************************************
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


## AE_Conv
##***********************************************************************************************************
class AE_Down(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_channels, out_channels):
        super(AE_Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AE_Up(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_channels, out_channels):
        super(AE_Up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
        