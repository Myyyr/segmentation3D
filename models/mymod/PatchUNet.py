import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D
import random as rd
from models.mymod.UNet import UNet


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
        if self.training:
            return self.unet(inp)
        else:
            IDXpatch = Patch(h,w,d,self.ps_h, self.ps_w, self.ps_d)
            nh, nw, nd = IDXpatch.nh, IDXpatch.nw, IDXpatch.nd
            out = torch.zeros(bs, self.n_classes, h, w, d)
            count = torch.zeros(bs, self.n_classes, h, w, d)
            for i in range(nh):
                for j in range(nw):
                    for k in range(nd):
                        x,y,z = IDXpatch(i,j,k)
                        count[:,:,x:(x+self.ps_h),y:(y+self.ps_w),z:(z+self.ps_d)] += 1
                        patch_ijk = inp[:,:,x:(x+self.ps_h),y:(y+self.ps_w),z:(z+self.ps_d)]
                        out_ijk = self.unet(patch_ijk)
                        out[...,x:(x+self.ps_h),y:(y+self.ps_w),z:(z+self.ps_d)] += out_ijk
            out = out/count
            return out

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
            s = int((self.ps_h - h%self.ps_h))
            self.ds_h = [0 for i in range(self.nh)] 
            while sum(self.ds_h) < s:
                for i in range(1, self.nh):
                    if sum(self.ds_h) < s:
                        self.ds_h[i] += 1
        if (w%self.ps_w!=0):
            self.nw += 1
            s = int((self.ps_w - w%self.ps_w))
            self.ds_w = [0 for i in range(self.nw)] 
            while sum(self.ds_w) < s:
                for i in range(1, self.nw):
                    if sum(self.ds_w) < s:
                        self.ds_w[i] += 1
        if (d%self.ps_d!=0):
            self.nd += 1
            s = int((self.ps_d - d%self.ps_d))
            self.ds_d = [0 for i in range(self.nd)] 
            while sum(self.ds_d) < s:
                for i in range(1, self.nd):
                    if sum(self.ds_d) < s:
                        self.ds_d[i] += 1        

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
        


def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size












