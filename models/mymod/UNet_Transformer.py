import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D
from models.mymod.transTools import Trans2D
from models.networks_other import init_weights

class UNetTransformer(nn.Module):

    def __init__(self, filters, n_classes=2, in_channels=1, n_heads=1, dim='2d', bn = True):
        super(UNetTransformer, self).__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.filters = filters
        self.n_heads = n_heads
        self.UNetConv = {'2d':UNetConv2D, '3d':UNetConv3D}[self.dim]
        self.UNetUpLayer = {'2d':UnetUp2D, '3d':UnetUp3D}[self.dim]
        self.maxpool = {'2d':nn.MaxPool2d, '3d':nn.MaxPool3d}[self.dim]
        self.final_layer = {'2d':nn.Conv2d, '3d':nn.Conv3d}[self.dim]
        self.trans_layer = {'2d':Trans2D}[self.dim]
        
        # encoder
        self.conv1 = self.UNetConv(self.in_channels, filters[0], bn=bn)
        self.maxpool1 = self.maxpool(kernel_size=2)

        self.conv2 = self.UNetConv(filters[0], filters[1], bn=bn)
        self.maxpool2 = self.maxpool(kernel_size=2)

        self.conv3 = self.UNetConv(filters[1], filters[2], bn=bn)
        self.maxpool3 = self.maxpool(kernel_size=2)

        self.conv4 = self.UNetConv(filters[2], filters[3], bn=bn)
        self.maxpool4 = self.maxpool(kernel_size=2)

        self.center = self.UNetConv(filters[3], filters[4], bn=bn)
        self.transformer = self.trans_layer(filters[4],self.n_heads)

        # upsampling
        self.up_concat4 = self.UNetUpLayer(filters[4], filters[3], bn=bn)
        self.up_concat3 = self.UNetUpLayer(filters[3], filters[2], bn=bn)
        self.up_concat2 = self.UNetUpLayer(filters[2], filters[1], bn=bn)
        self.up_concat1 = self.UNetUpLayer(filters[1], filters[0], bn=bn)

        # final conv (without any concat)
        self.final = self.final_layer(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, self.final_layer):
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

