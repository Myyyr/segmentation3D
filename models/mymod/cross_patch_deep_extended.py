import torch.nn as nn
import torch.nn.functional as F
import torch
from models.mymod.utils import UNetConv2D, UNetConv3D, UnetUp2D, UnetUp3D, PositionalEncodingPermute3D
from models.mymod.transTools import PositionalEncoding, CrossAttention
from models.networks_other import init_weights
import numpy as np
from einops import rearrange, repeat


class SimpleTransEncoder(nn.Module):
    def __init__(self, d_model, n_heads=8, n_layer=1):
        super(SimpleTransEncoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layer = n_layer

        # Transformer for self attention
        self.p_enc_3d = PositionalEncodingPermute3D(d_model)
        trans_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads)
        self.self_trans = nn.TransformerEncoder(trans_layer, self.n_layer)

        self.PE = None

    def forward(self, x, pos=None):
        bs,c,h,w,d = x.shape

        if self.PE == None:
            bs, c, Sh, Sw, Sd = x.shape
            z = torch.zeros((bs,c,(Sh*3),(Sw*3),(Sd*4))).float().cuda()
            self.PE = self.p_enc_3d(z)
            del z

        x = self.apply_positional_encoding(pos, self.PE, x)

        Y = rearrange(x, 'b c h w d -> b (h w d) c')
        del x

        ## Permutation
        Y = rearrange(Y, 'b n d -> n b d') # seq, bs, d_model # for pytorch tranformer layer

        ## Transformer
        Y = self.self_trans(Y)

        ## Permutation
        # Y = Y.permute(1,0,2)
        Y = rearrange(Y, 'n b d -> b n d')
        Y = rearrange(Y, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

        return Y

    def apply_positional_encoding(self, pos, pe, x):        
        bs, c, h, w, d = x.shape   
        ret = torch.zeros(x.shape).float().cuda() 
        fh = 192//h    
        fw = 192//w
        fd = 48//d
        for i in range(bs):
            a,b,c = (pos[i,0]//fh).item(), (pos[i,1]//fw).item(), (pos[i,2]//fd).item()
            ret[i, ...] = x[i,...] + pe[i, :, a:a+h, b:b+w, c:c+d]
        return x



class SelfTransEncoder(nn.Module):
    """docstring for SelfTransEncoder"""
    def __init__(self, filters = [16, 32, 64, 128, 256, 512], use_trans = [1,1,1,1,1,1], in_channels=1, n_sheads=8, bn = True, n_strans=6):
        super(SelfTransEncoder, self).__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.n_sheads = n_sheads
        self.use_trans = use_trans

        
        # CNN encoder
        # self.first_conv = nn.Conv3d(self.in_channels, filters[0], 1)
        # self.conv1 = UNetConv3D(filters[0], filters[0], bn=bn)
        self.conv1 = UNetConv3D(self.in_channels, filters[0], bn=bn)
        if use_trans[0]:
            self.trans1 = SimpleTransEncoder(filters[0], n_sheads, n_strans)

        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = UNetConv3D(filters[0], filters[1], bn=bn)
        if use_trans[1]:
            self.trans2 = SimpleTransEncoder(filters[1], n_sheads, n_strans)

        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = UNetConv3D(filters[1], filters[2], bn=bn)
        if use_trans[2]:
            self.trans3 = SimpleTransEncoder(filters[2], n_sheads, n_strans)

        self.maxpool4 = nn.MaxPool3d(kernel_size=2)
        self.conv4 = UNetConv3D(filters[2], filters[3], bn=bn)
        if use_trans[3]:
            self.trans4 = SimpleTransEncoder(filters[3], n_sheads, n_strans)

        self.maxpool5 = nn.MaxPool3d(kernel_size=2)
        self.conv5 = UNetConv3D(filters[3], filters[4], bn=bn)
        if use_trans[4]:
            self.trans5 = SimpleTransEncoder(filters[4], n_sheads, n_strans)

        self.maxpool6 = nn.MaxPool3d(kernel_size=(2,2,1))
        self.conv6 = UNetConv3D(filters[4], filters[5], bn=bn)
        if use_trans[5]:
            self.trans6 = SimpleTransEncoder(filters[5], n_sheads, n_strans)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')


    def forward(self, X, ret_skip=True, pos=None):
        skip1 = self.conv1(X)
        if self.use_trans[0]:
            skip1 = self.trans1(x, pos)
        del X

        skip2 = self.maxpool2(skip1)
        if not ret_skip: del skip1
        skip2 = self.conv2(skip2)
        if self.use_trans[1]:
            skip2 = self.trans2(skip2, pos)

        skip3 = self.maxpool3(skip2)
        if not ret_skip: del skip2
        skip3 = self.conv3(skip3)
        if self.use_trans[2]:
            skip3 = self.trans3(skip3, pos)

        skip4 = self.maxpool4(skip3)
        if not ret_skip: del skip3
        skip4 = self.conv4(skip4)
        if self.use_trans[3]:
            skip4 = self.trans4(skip4, pos)

        skip5 = self.maxpool5(skip4)
        if not ret_skip: del skip4
        skip5 = self.conv5(skip5)
        if self.use_trans[4]:
            skip5 = self.trans5(skip5, pos)

        skip6 = self.maxpool6(skip5)
        if not ret_skip: del skip5
        skip6 = self.conv6(skip6)
        if self.use_trans[5]:
            skip6 = self.trans6(skip6, pos)


        if ret_skip: 
            return skip6, (skip1, skip2, skip3, skip4, skip5)
        return skip6


class CrossPatch3DTr(nn.Module):

    def __init__(self, filters = [16, 32, 64, 128, 256, 512], use_trans = [1,1,1,1,1,1],n_classes=14, in_channels=1, n_cheads=2, n_sheads=8, bn = True, up_mode='deconv', n_strans=6, do_cross=False):
        super(CrossPatch3DTr, self).__init__()
        self.PE = None

        self.in_channels = in_channels
        self.filters = filters
        self.n_sheads = n_sheads
        self.use_trans = use_trans
        self.do_cross = do_cross

        # CNN + Trans encoder
        self.encoder = SelfTransEncoder(filters=filters, use_trans=use_trans, in_channels=in_channels, n_sheads=n_sheads, bn=bn, n_strans=n_strans)


        # Transformer for cross attention
        self.p_enc_3d = PositionalEncodingPermute3D(filters[-1])
        self.cross_trans = CrossAttention(filters[-1], n_cheads)


        # CNN decoder
        ## Decode like 3D UNet
        self.up_concat5 = UnetUp3D(filters[5], filters[4], bn=bn, up_mode=up_mode,kernel=(2,2,1), stride=(2,2,1))
        self.up_concat4 = UnetUp3D(filters[4], filters[3], bn=bn, up_mode=up_mode)
        self.up_concat3 = UnetUp3D(filters[3], filters[2], bn=bn, up_mode=up_mode)
        self.up_concat2 = UnetUp3D(filters[2], filters[1], bn=bn, up_mode=up_mode)
        self.up_concat1 = UnetUp3D(filters[1], filters[0], bn=bn, up_mode=up_mode)
        

        self.final_conv = nn.Conv3d(filters[0], n_classes, 1)

        # Deep Supervision
        self.ds_cv1 = nn.Conv3d(filters[4], n_classes, 1)
        self.ds_cv2 = nn.Conv3d(filters[3], n_classes, 1)
        self.ds_cv3 = nn.Conv3d(filters[2], n_classes, 1)
        self.ds_cv4 = nn.Conv3d(filters[1], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def apply_positional_encoding(self, pos, pe, x):        
        bs, c, h, w, d = x.shape   
        ret = torch.zeros(x.shape).float().cuda()   
        fh = 192//h    
        fw = 192//w
        fd = 48//d
        for i in range(bs):
            a,b,c = (pos[i,0]//fh).item(), (pos[i,1]//fw).item(), (pos[i,2]//fd).item()
            ret[i, ...] = x[i,...] + pe[i, :, a:a+h, b:b+w, c:c+d]
        return x

    def forward(self, X, pos, val=False):
        if self.do_cross:      
            R = X[:,:,0 ,...]
            A = X[:,:,1:,...]
            encoder_grad = torch.no_grad
        else:
            R = X
            encoder_grad = torch.enable_grad
        if val:
            encoder_grad = torch.no_grad

        # Create PE
        ## Be carefull here if you change region size or bottleneck qpatial size you
        ## have to adapt the positionnal enccoding size.
        ## (Sh, Sw, Sd) is the spatial size of the bottleneck.
        ## (3,3,4) is the image size divided by the patch size.
        Sh,Sw,Sd = (6,6,3)
        c = self.filters[-1]
        bs = X.shape[0]
        if self.PE==None:
            z = torch.zeros((bs,c,(Sh*3),(Sw*3),(Sd*4))).float().cuda()
            self.PE = self.p_enc_3d(z)
        posR = pos[:,0 ,...]
        posA = pos[:,1:,...]

        # Encode the interest region
        with encoder_grad():
            R, S = self.encoder(R, True, posR)
        skip1, skip2, skip3, skip4, skip5 = S
        bs, c, h, w, d = skip5.shape
        c = c*2
        # h = h//2
        # w = w//2
        # d = d//2
        h = Sh
        w = Sw
        d = Sd


        if self.do_cross:
            R = self.apply_positional_encoding(posR, self.PE, R)
            R = rearrange(R, 'b c h w d -> b (h w d) c')

        
        
            # Encode all regions with no gradient
            YA = []
            bs,_,na,_,_,_ = A.shape
            with torch.no_grad():
                for ra in range(na):
                    enc = self.encoder(A[:,:,ra,...], False, posA[:,ra,...])

                    # Positional encodding
                    enc = self.apply_positional_encoding(posA[:,ra,...], self.PE, enc)
                    enc = rearrange(enc, 'b c h w d -> b (h w d) c')
                    YA.append(enc)

            # Concatenate all feature maps
            A = torch.cat([R] + YA, 1)
            del YA, X

            rseq = R.shape[1]

            # Cross attention
            Z = self.cross_trans(A, rseq)
            del A
            
            # Decoder
            ## Permute and Reshape
            Z = rearrange(Z, ('b n c -> b c n'))
            Z = rearrange(Z, ('b c (h w d) -> b c h w d'), h=h, w=w, d=d)
            

        else:
            # bs,_,na,_,_,_ = A.shape
            na = 3*3*4
            R = self.apply_positional_encoding(posR, self.PE, R)
            R = rearrange(R, 'b c h w d -> b (h w d) c')
            YA = [R for r in range(na)]
            A = torch.cat([R] + YA, 1)
            del YA, X

            rseq = R.shape[1]

            # Cross attention
            Z = self.cross_trans(A, rseq)
            del A
            
            # Decoder
            ## Permute and Reshape
            Z = rearrange(Z, ('b n c -> b c n'))
            # print(Z.shape, h, w, d)
            Z = rearrange(Z, ('b c (h w d) -> b c h w d'), h=h, w=w, d=d)
        

        ## Up, skip, conv and ds
        Z = self.up_concat5(skip5, Z)
        ds1 = self.ds_cv1(Z)
        del skip5
        Z = self.up_concat4(skip4, Z)
        ds2 = self.ds_cv2(Z)
        del skip4
        Z = self.up_concat3(skip3, Z)
        ds3 = self.ds_cv3(Z)
        del skip3
        Z = self.up_concat2(skip2, Z)
        ds4 = self.ds_cv4(Z)
        del skip2
        Z = self.up_concat1(skip1, Z)
        del skip1

        ## get prediction with final layer
        Z = self.final_conv(Z)
        return [Z, ds4, ds3, ds2, ds1]



def convert_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.2f %s" % (size, x)
        size /= 1024.0

    return size

