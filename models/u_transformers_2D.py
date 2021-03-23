import math
import torch.nn as nn
from models.unetUtils import UnetConv2, unetUp
import torch.nn.functional as F
from models.networks_other import init_weights
import torch
from models.transTools import trans

class u_transformers_2D(nn.Module):

    def __init__(self,filters, trans_shape, n_classes=21, is_deconv=True, in_channels=4, is_batchnorm=True):
        super(u_transformers_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        # self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]
        self.filters = filters

        # downsampling
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)
        self.trans = trans(*trans_shape)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = unetUp(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = unetUp(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = unetUp(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)



        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, X):

        conv1 = self.conv1(X)
        del X
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        # print('conv4.shape:',conv4.shape)
        maxpool4 = self.maxpool4(conv4)
        # print('maxpool4.shape:',maxpool4.shape)



        # center = self.center(maxpool4)
        center, q, k, v = self.trans(self.center(maxpool4))
        del q, k ,v

        up4 = self.up_concat4(conv4, center)
        # print('up4.shape:',up4.shape)
        up3 = self.up_concat3(conv3, up4)
        # print('up3.shape:',up3.shape)
        up2 = self.up_concat2(conv2, up3)
        # print('up2.shape:',up2.shape)
        up1 = self.up_concat1(conv1, up2)
        # print('up1.shape:',up1.shape)
# 
        # print("||down/up|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||down/up|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))

        del maxpool1, maxpool2, maxpool3, maxpool4, center
        del conv1,conv2,conv3,conv4
        del up4, up3, up2
        # print("||del maxpool|| memory :",convert_bytes(torch.cuda.max_memory_allocated()))
        # print("||del maxpool|| cur memory :", convert_bytes(torch.cuda.memory_allocated()))

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












