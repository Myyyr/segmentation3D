import torch.nn as nn
import torch
from models.networks_other import init_weights


class UNetConv2D(nn.Module):
    def __init__(self, in_size, out_size, kernel=(3,3), pad=(1,1), stride=(1,1)):
        super(UNetConv2D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride, pad),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel, 1, pad),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace=True),)

        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UNetConv3D(nn.Module):
    def __init__(self, in_size, out_size, kernel=(3,3,3), pad=(1,1,1), stride=(1,1,1)):
        super(UNetConv3D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel, stride, pad),
                                   nn.BatchNorm3d(out_size),
                                   nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel, 1, pad),
                                   nn.BatchNorm3d(out_size),
                                   nn.ReLU(inplace=True),)
        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp2D(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp2D, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv0 = nn.Conv2d(in_size, out_size, kernel_size=(2,2))
        self.conv1 = UNetConv2D(in_size, out_size)

        #initialise the blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.conv0(nn.functional.pad(self.up(inputs2), (1,0,1,0)))
        return self.conv1(torch.cat([inputs1, outputs2], 1))


class UnetUp3D(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp3D, self).__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.conv0 = nn.Conv3d(in_size, out_size, kernel_size=(2,2,2))
        self.conv1 = UNetConv3D(in_size, out_size)

        #initialise the blocks
        for m in self.children():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
    def forward(self, inputs1, inputs2):
        # outputs2 = self.conv0(self.up(inputs2))
        outputs2 = self.conv0(nn.functional.pad(self.up(inputs2), (1,0,1,0,1,0)))
        return self.conv1(torch.cat([inputs1, outputs2], 1))