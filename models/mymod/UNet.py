import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D
import random as rd

class UNet(nn.Module):

    def __init__(self, filters, n_classes=2, in_channels=1, dim='2d'):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.filters = filters
        self.UNetConv = {'2d':UNetConv2D, '3d':UNetConv3D}[self.dim]
        self.UNetUpLayer = {'2d':UnetUp2D, '3d':UnetUp3D}[self.dim]
        self.maxpool = {'2d':nn.MaxPool2d, '3d':nn.MaxPool3d}[self.dim]
        self.final_layer = {'2d':nn.Conv2d, '3d':nn.Conv3d}[self.dim]
        # encoder
        self.conv1 = self.UNetConv(self.in_channels, filters[0])
        self.maxpool1 = self.maxpool(kernel_size=2)

        self.conv2 = self.UNetConv(filters[0], filters[1])
        self.maxpool2 = self.maxpool(kernel_size=2)

        self.conv3 = self.UNetConv(filters[1], filters[2])
        self.maxpool3 = self.maxpool(kernel_size=2)

        self.conv4 = self.UNetConv(filters[2], filters[3])
        self.maxpool4 = self.maxpool(kernel_size=2)

        self.center = self.UNetConv(filters[3], filters[4])

        # upsampling
        self.up_concat4 = self.UNetUpLayer(filters[4], filters[3])
        self.up_concat3 = self.UNetUpLayer(filters[3], filters[2])
        self.up_concat2 = self.UNetUpLayer(filters[2], filters[1])
        self.up_concat1 = self.UNetUpLayer(filters[1], filters[0])

        # final conv (without any concat)
        self.final = self.final_layer(filters[0], n_classes, 1)


    def forward(self, X, mode = None):
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

"""
class Patched3DUNet(nn.Module):
    def __init__(self, patch_size, filters, n_classes=2, in_channels=1):
        super(Patched3DUNet, self).__init__()
        self.in_channels = in_channels
        self.dim = '3d'
        self.filters = filters
        # self.patch_size = patch_size
        self.ps_h, self.ps_w, self.ps_d = patch_size
        self.n_classes = n_classes

        self.unet = UNet(filters=self.filters, n_classes=self.n_classes, in_channels=self.in_channels, dim=self.dim)



    def forward(self, inp, mode = 'train'):
        bs, c, h, w, d = inp.shape
        # if mode == 'train':
        if self.training:
            return self.unet(inp)
        else:
            IDXpatch = Patch(h,w,d,self.ps_h, self.ps_w, self.ps_d)
            nh, nw, nd = IDXpatch.nh, IDXpatch.nw, IDXpatch.nd
            print(nh,nw,nd)
            out = torch.zeros(bs, self.n_classes, h, w, d)
            count = torch.zeros(bs, self.n_classes, h, w, d)
            for i in range(nh):
                for j in range(nw):
                    for k in range(nd):
                        # x,y,z = i*self.ps_h, j*self.ps_w, k*self.ps_d
                        x,y,z = IDXpatch(i,j,k)
                        # if x + self.ps_h > h: 
                        #     sup_x = (x, h - self.ps_h)
                        #     x = h - self.ps_h
                        # if y + self.ps_w > w: 
                        #     sup_y = (y, w - self.ps_w)
                        #     y = w - self.ps_w
                        # if z + self.ps_d > d: 
                        #     sup_z = (z, d - self.ps_d)
                        #     z = d - self.ps_d
                        count[:,:,x:(x+self.ps_h),y:(y+self.ps_w),z:(z+self.ps_d)] += 1
                        patch_ijk = inp[:,:,x:(x+self.ps_h),y:(y+self.ps_w),z:(z+self.ps_d)]
                        print(x,y,z,patch_ijk.shape)
                        out_ijk = self.unet(patch_ijk)
                        out[...,x:(x+self.ps_h),y:(y+self.ps_w),z:(z+self.ps_d)] = out_ijk
            out = out/count
            return out, count

class Patch():
    def __init__(self, h, w, d, ps_h, ps_w, ps_d):
        self.ps_h = ps_h
        self.ps_w = ps_w
        self.ps_d = ps_d

        self.h = h
        self.w = w
        self.d = d

        self.nh = int(h//self.ps_h)
        self.nw = int(w//self.ps_w)
        self.nd = int(d//self.ps_d)

        if (h%self.ps_h!=0):
            self.nh += 1
            s = int((self.ps_h - h%self.ps_h)/self.nh)
            ms = int((self.ps_h - h%self.ps_h)%self.nh)
            self.ds_h = [0] + [s* for a in range(1, self.nh-1)] + [0]
            self.ds_h[self.ds_d]
        if (w%self.ps_w!=0):
            self.nw += 1
            self.ds_w = [0] + [s* for a in range(1, self.nw-1)] + [0]
            self.ds_h[self.ds_d]
        if (d%self.ps_d!=0):
            self.nd += 1
            self.ds_d = [0] + [s* for a in range(1, self.nd-1)] + [0]
            self.ds_h[self.ds_d]

    def __call__(self, i,j,k):
        x,y,z = i*self.ps_h, j*self.ps_w, k*self.ps_d

        if (self.h%self.ps_h!=0):
            x -= sum(self.ds_h[:(i+1)])
            # x = (i-1)*self.ps_h + self.h%self.ps_h
        if (self.w%self.ps_w!=0):
            y -= sum(self.ds_w[:(j+1)])
            # y = (j-1)*self.ps_w + self.w%self.ps_w
        if (self.d%self.ps_d!=0):
            z -= sum(self.ds_d[:(k+1)])
            # z = (k-1)*self.ps_d + self.d%self.ps_d

        return (x,y,z)
        
"""

def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size












