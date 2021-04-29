import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D
from models.mymod.transTools import PositionalEncoding, CrossAttention
from models.networks_other import init_weights
import numpy as np

class SelfTransEncoder(nn.Module):
    """docstring for SelfTransEncoder"""
    def __init__(self, filters = [16, 32, 64, 128], patch_size = [2,2,2], d_model = 1024, in_channels=1, n_sheads=8, bn = True, n_strans=6):
        super(SelfTransEncoder, self).__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.n_sheads = n_sheads
        self.d_model = d_model
        self.patch_size = patch_size

        
        # CNN encoder
        self.first_conv = nn.Conv3d(self.in_channels, filters[0], 1)
        self.conv1 = UNetConv3D(filters[0], filters[0], bn=bn)

        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = UNetConv3D(filters[0], filters[1], bn=bn)

        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = UNetConv3D(filters[1], filters[2], bn=bn)

        self.maxpool4 = nn.MaxPool3d(kernel_size=2)
        self.conv4 = UNetConv3D(filters[2], filters[3], bn=bn)

        
        # Transformer for self attention
        self.before_d_model = filters[3]*np.prod(self.patch_size)
        self.linear = nn.Linear(self.before_d_model, self.d_model)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=0.1, max_len = 1000)
        trans_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_sheads)
        self.self_trans = nn.TransformerEncoder(trans_layer, n_strans)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
        

    def forward(self, X, ret_skip=True):
        
        # CNN Encoder
        skip1 = self.first_conv(X)
        skip1 = self.conv1(skip1)
        del X

        skip2 = self.maxpool2(skip1)
        if not ret_skip: del skip1
        skip2 = self.conv2(skip2)

        skip3 = self.maxpool3(skip2)
        if not ret_skip: del skip2
        skip3 = self.conv3(skip3)

        skip4 = self.maxpool4(skip3)
        if not ret_skip: del skip3
        skip4 = self.conv3(skip3)

        # Transformer for self attention
        ## Patch, Reshapping
        bs,c,h,w,d = Y.shape
        s1, s2, s3 = self.patch_size
        s = s1*s2*s3
        n_seq = int(h*w*d/s)
        # print(Y.shape)
        Y = torch.reshape(skip4, (bs, c, n_seq, s1, s2, s3))
        Y = torch.reshape(Y, (bs, c, n_seq, s))
        Y = Y.permute(0,2,1,3) # bs, seq, c, s
        Y = torch.reshape(Y, (bs,n_seq,self.before_d_model))
        
        ## Linear projection
        Y = self.linear(Y)
        if not ret_skip: del skip4

        ## Positional encodding
        Y = self.positional_encoder(Y)

        ## Permutation
        Y = Y.permute(1,0,2) # seq, bs, bef_dmodel # for pytorch tranformer layer

        ## Transformer
        Y = self.self_trans(Y)

        ## Permutation
        Y = Y.permute(1,0,2)

        if ret_skip: 
            return Y, (skip1, skip2, skip3, skip4)
        return Y


class CrossPatch3DTr(nn.Module):

    def __init__(self, filters = [16, 32, 64, 128], patch_size = [2,2,2], d_model = 1024,n_classes=14, in_channels=1, n_cheads=2, n_sheads=8, bn = True, up_mode='deconv', n_strans=6):
        super(CrossPatch3DTr, self).__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.n_sheads = n_sheads
        self.d_model = d_model
        self.patch_size = patch_size


        # CNN + Trans encoder
        self.encoder = SelfTransEncoder(filters=filters, patch_size=patch_size, d_model=d_model, in_channels=in_channels, n_sheads=n_sheads, bn=bn, n_strans=n_strans)


        # Transformer for cross attention
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=0.1, max_len = 20000)
        self.cross_trans = CrossAttention(self.d_model, n_cheads)


        # CNN decoder 
        self.before_d_model = filters[3]*np.prod(self.patch_size)
        ## Rescale progressively feature map from cross attention
        a = int(self.before_d_model/self.patch_size[0])
        b = int(a/self.patch_size[1])
        c = int(b/self.patch_size[2])
        self.center = nn.Sequential(nn.ConvTranspose3d(self.d_model, a, 2, stride=2),
                                    nn.Conv3d(a,b, 3, padding=1),
                                    nn.Conv3d(b,c, 3, padding=1))

        ## Decode like 3D UNet
        self.up_concat3 = UnetUp3D(filters[3], filters[2], bn=bn, up_mode=up_mode)
        self.up_concat2 = UnetUp3D(filters[2], filters[1], bn=bn, up_mode=up_mode)
        self.up_concat1 = UnetUp3D(filters[1], filters[0], bn=bn, up_mode=up_mode)
        self.final_conv = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')


    def forward(self, X):        
        R = X[:,:,0 ,...]
        A = X[:,:,1:,...]

        # Encode the interest region
        R, S = self.encoder(R, True)
        skip1, skip2, skip3, skip4 = S

        # Encode all regions with no gradient
        YA = []
        bs,_,na,_,_,_ = A.shape
        with torch.no_grad():
            for ra in range(na):
                YA.append(self.encoder(A[:,:,ra,...], False))

        # Concatenate all feature maps
        A = torch.cat([R] + YA, 1)
        del YA, X

        # Positional encodding
        A = self.positional_encoder(A)
        rseq = R.shape[1]
        del R

        # Cross attention
        # print(A.shape)
        Z = self.cross_trans(A, rseq)
        del A
        
        # Decoder
        ## Permute and Reshape
        _, c, h, w, d = skip3.shape
        Z = Z.permute(0,2,1)
        # print('skip3.shape', skip3.shape)
        # print((bs, self.d_model, int(h/self.patch_size[0]), int(h/self.patch_size[1]), int(h/self.patch_size[2])))
        Z = torch.reshape(Z, (bs, self.d_model, int(h/self.patch_size[0]), int(h/self.patch_size[1]), int(h/self.patch_size[2])))

        ## Progressively rescale featue map Z
        print(Z.shape)
        Z = self.center(Z)
        print(Z.shape)
        exit(0)

        ## Up, skip and conv
        Z = self.up_concat3(skip3, Z)
        del skip3
        Z = self.up_concat2(skip2, Z)
        del skip2
        Z = self.up_concat1(skip1, Z)
        del skip1

        ## get prediction with final layer
        Z = self.final_conv(Z)
        return Z



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size

