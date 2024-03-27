import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY

class Atten(nn.Module):
    def __init__(self, input_channels):
        super(Atten, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_channels // 4, input_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.global_pool(inputs)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        x = x.view(-1,inputs.size(1), 1, 1)
        # print(x.shape)
        x = inputs * x
        return x


class Nor_Conv_block(nn.Module):
    def __init__(self, output_channels=200, kernel_size=3, use_bias=False):
        super(Nor_Conv_block, self).__init__()
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,
                                padding='same', bias=use_bias)
        self.conv_2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,
                                padding='same', bias=use_bias)
        self.conv_3 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size,
                                padding='same', bias=use_bias)
        self.bn_1 = nn.BatchNorm2d(output_channels, momentum=0.8)
        self.bn_2 = nn.BatchNorm2d(output_channels, momentum=0.8)
        self.bn_3 = nn.BatchNorm2d(output_channels, momentum=0.8)

    def forward(self, X):
        X = self.conv_1(X)
        X = self.bn_1(X)
        X = F.relu(X)
        X = self.conv_2(X)
        X = self.bn_2(X)
        X = F.relu(X)
        X = self.conv_3(X)
        X = self.bn_3(X)
        X = F.relu(X)
        return X

    def get_config(self):
        return {
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size
        }



class Rec_Conv_block(nn.Module):
    def __init__(self, input_channels=3, output_channels=200, kernel_size=3, use_bias=False):
        super(Rec_Conv_block, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.conv_0 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, bias=use_bias)
        self.conv_1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=use_bias)
        self.conv_2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=use_bias)
        self.conv_3 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=use_bias)
        self.conv_4 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=use_bias)
        self.bn_0 = nn.BatchNorm2d(output_channels, momentum=0.95)
        self.bn_1 = nn.BatchNorm2d(output_channels, momentum=0.95)
        self.bn_2 = nn.BatchNorm2d(output_channels, momentum=0.95)
        self.bn_3 = nn.BatchNorm2d(output_channels, momentum=0.95)
        self.bn_4 = nn.BatchNorm2d(output_channels, momentum=0.95)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        x = self.conv_0(x)
        x = self.bn_0(x)
        x = F.relu(x)
        # print(x.shape)

        X = x.clone()
        X = self.conv_1(X)
        X = self.bn_1(X)
        X = F.relu(X)
        X = self.conv_2(X + x)
        X = self.bn_2(X)
        X = F.relu(X)
        X = self.conv_3(X + x)
        X = self.bn_3(X)
        X = F.relu(X)
        return X + x


Conv_block = Rec_Conv_block


class DWT_downsampling(nn.Module):
    def __init__(self):
        super(DWT_downsampling, self).__init__()

    def forward(self, x):
        x1 = x[:, :, 0::2, 0::2]  # x(2i−1, 2j−1)
        x2 = x[:, :, 1::2, 0::2]  # x(2i, 2j-1)
        x3 = x[:, :, 0::2, 1::2]  # x(2i−1, 2j)
        x4 = x[:, :, 1::2, 1::2]  # x(2i, 2j)

        x_LL = (x1 + x2 + x3 + x4) / 4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4

        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1) 



class IWT_upsampling(nn.Module):
    def __init__(self):
        super(IWT_upsampling, self).__init__()

    def forward(self, x):
        shape = x.shape
        # print(shape)
        x_LL = x[:, 0:x.shape[1] // 4, :, :] * 4
        x_LH = x[:, x.shape[1] // 4:x.shape[1] // 4 * 2, :, :]
        x_HL = x[:, x.shape[1] // 4 * 2:x.shape[1] // 4 * 3, :, :]
        x_HH = x[:, x.shape[1] // 4 * 3:, :, :]

        x1 = (x_LL - x_LH - x_HL + x_HH) / 4
        x2 = (x_LL - x_LH + x_HL - x_HH) / 4
        x3 = (x_LL + x_LH - x_HL - x_HH) / 4
        x4 = (x_LL + x_LH + x_HL + x_HH) / 4
        # print(x1.shape)

        y1 = torch.cat([x1, x3], dim=3)
        y2 = torch.cat([x2, x4], dim=3)
        # print(y1.shape)
        return (torch.cat([y1, y2], dim=2))



# class MWCNN(nn.Module):
#     def __init__(self):
#         super(MWCNN, self).__init__()
#         self.cb_1 = Conv_block(output_channels=64)
#         self.dwt_1 = DWT_downsampling()
#         self.cb_2 = Conv_block(output_channels=128)
#         self.dwt_2 = DWT_downsampling()
#         self.cb_3 = Conv_block(output_channels=256)
#         self.dwt_3 = DWT_downsampling()
#         self.cb_5 = Conv_block(output_channels=512)
#         self.bn_5_1 = nn.BatchNorm2d(512, momentum=0.8)
#         self.bn_5_2 = nn.BatchNorm2d(512, momentum=0.8)
#         self.bn_5_3 = nn.BatchNorm2d(512, momentum=0.8)
#         self.bn_5_4 = nn.BatchNorm2d(512, momentum=0.8)
#         self.conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False)
#         self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False)
#         self.conv_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False)
#         self.conv_5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False)
#         self.up = IWT_upsampling()
#         self.up_conv_1 = Conv_block(output_channels=256)
#         self.up_conv_2 = Conv_block(output_channels=512)
#         self.up_conv_3 = Conv_block(output_channels=256)
#         self.up_conv_4 = Conv_block(output_channels=256)
#         self.out_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False)
#         self.atten = Atten(input_channels=256)

#     def forward(self, input_L, input_R, input_img):
#         input = torch.cat([input_R, input_L], dim=1)
#         cb_1 = self.cb_1(input)
#         dwt_1 = self.dwt_1(cb_1)

#         cb_2 = self.cb_2(dwt_1)
#         dwt_2 = self.dwt_2(cb_2)

#         cb_3 = self.cb_3(dwt_2)
#         dwt_3 = self.dwt_3(cb_3)

#         cb_5 = self.cb_5(dwt_3)
#         cb_5 = self.bn_5_1(cb_5)
#         cb_5 = F.relu(cb_5)
#         cb_5 = self.bn_5_2(cb_5)
#         cb_5 = F.relu(cb_5)
#         cb_5 = self.bn_5_3(cb_5)
#         cb_5 = F.relu(cb_5)
#         cb_5 = self.bn_5_4(cb_5)
#         cb_5 = F.relu(cb_5)
#         cb_5 = self.conv_5_1(cb_5)
#         cb_5 = self.conv_5_2(cb_5)
#         cb_5 = self.conv_5_3(cb_5)
#         cb_5 = self.conv_5_4(cb_5)

#         up = self.up(cb_5)
#         up = self.up_conv_1(up + cb_3)
#         up = self.up_conv_2(up)
#         up = self.up_conv_3(up)
#         up = self.up_conv_4(up)
#         out = self.out_conv(self.atten(up))

#         return torch.sigmoid(out)



@ARCH_REGISTRY.register()
class R2MWCNN(nn.Module):
    def __init__(self, use_bias=False):
        super(R2MWCNN, self).__init__()
        self.relu = F.relu
        self.dwt = DWT_downsampling()

        # MWCNN
        self.cb_11 = Conv_block(input_channels = 3, output_channels=32)
        self.cb_12 = Conv_block(input_channels = 32, output_channels=32)
        self.cb_2 = Conv_block(input_channels = 128, output_channels=64)
        self.cb_3 = Conv_block(input_channels = 256, output_channels=64)
        self.cb_51 = Conv_block(input_channels = 256, output_channels=128)
        self.cb_52 = Conv_block(input_channels = 128, output_channels=128)

        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn_5 = nn.BatchNorm2d(256, momentum=0.8)

        self.up = IWT_upsampling()

        self.up_conv_1 = Conv_block(input_channels = 64, output_channels=128)
        self.up_conv_2 = Conv_block(input_channels = 64, output_channels=128)
        self.up_conv_3 = Conv_block(input_channels = 32, output_channels=128)

        self.out_conv_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.out_conv_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.out_conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.bn_out_3 = nn.BatchNorm2d(128, momentum=0.8)
        self.out_conv_out1 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.out_conv_out2 = nn.Conv2d(in_channels=128, out_channels=3,kernel_size=1, stride=1, padding=0, bias=use_bias)

        self.atten = Atten(input_channels=128)


    def forward(self, input_img):
        cb_11 = self.cb_11(input_img)
        cb_12 = self.cb_12(cb_11) #32
        dwt_1 = self.dwt(cb_12)  #128

        cb_2 = self.cb_2(dwt_1) #64
        dwt_2 = self.dwt(cb_2)  #256

        cb_3 = self.cb_3(dwt_2) #64
        dwt_3 = self.dwt(cb_3)  #256

        cb_51 = self.cb_51(dwt_3)
        cb_52 = self.cb_52(cb_51) #128
        cb_5 = self.conv_5(cb_52) #256
        cb_5 = self.bn_5(cb_5)
        cb_5 = self.relu(cb_5)

        up1 = self.up(cb_5)  #64
        up1 = self.up_conv_1(up1 + cb_3) #128
        up1 = self.out_conv_1(up1) #256

        up2 = self.up(up1) #64
        up2 = self.up_conv_2(up2 + cb_2) #128
        up2 = self.out_conv_2(up2) #128

        up3 = self.up(up2) #32
        up3 = self.up_conv_3(up3 + cb_12) #128
        up3 = self.out_conv_3(up3)
        up3 = self.bn_out_3(up3)
        up3 = self.relu(up3)
        out = self.out_conv_out1(up3)
        out = self.out_conv_out2(self.atten(out))

        return torch.sigmoid(out), cb_5


        