import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D
from models.mymod.transTools import Trans2D
from models.networks_other import init_weights

class CrossPatch3DTr(nn.Module):

    def __init__(self, filters = [32, 64, 128], n_classes=14, in_channels=1, n_heads=8, dim='2d', bn = True, up_mode='biline'):
        super(CrossPatch3DTr, self).__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.filters = filters
        self.n_heads = n_heads

        
        # CNN encoder
        self.conv1 = UNetConv3D(self.in_channels, filters[0], bn=bn)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = UNetConv3D(filters[0], filters[1], bn=bn)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = UNetConv3D(filters[1], filters[2], bn=bn)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        
        # Transformer for self attention
        self.linear = nn.Linear(filters[2])


        # upsampling
        self.up_concat4 = UnetUp3D(filters[4], filters[3], bn=bn, up_mode=up_mode)
        self.up_concat3 = UnetUp3D(filters[3], filters[2], bn=bn, up_mode=up_mode)
        self.up_concat2 = UnetUp3D(filters[2], filters[1], bn=bn, up_mode=up_mode)
        self.up_concat1 = UnetUp3D(filters[1], filters[0], bn=bn, up_mode=up_mode)

        # final conv (without any concat)
        self.final = self.nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')


    def forward(self, X, mode=None):
        conv1 = self.conv1(X)
        del X
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center,_,_,_ = self.transformer(center)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        del maxpool1, maxpool2, maxpool3, maxpool4, center
        del conv1,conv2,conv3,conv4
        del up4, up3, up2

        final = self.final(up1)
        del up1
        
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size

