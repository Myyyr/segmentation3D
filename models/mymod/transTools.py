import torch
import torch.nn as nn
import math
from models.networks_other import init_weights


class CrossAttention(nn.Module):
    """docstring for CrossAttention"""

    # !! Maybe add dropout later !!
    def __init__(self, d_model, n_heads = 2):
        super(CrossAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads


        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)


        self.all_w = []
        for i in range(self.n_heads):
            wq = nn.Linear(self.d_model, self.d_model)
            wk = nn.Linear(self.d_model, self.d_model)
            wv = nn.Linear(self.d_model, self.d_model)
            self.all_w.append(nn.ModuleList([wq, wk, wv]))
        self.all_w = nn.ModuleList(self.all_w)

        self.wo = nn.Linear(self.d_model*self.n_heads, self.d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.feed_forward = nn.Linear(d_model, d_model)

        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')        


    def forward(self, X, rseq=512):
        # Normalization
        # print('###X',X.shape)
        X = self.norm1(X)
        # print('###X',X.shape)

        # Separation Region/FullImage into Xq / (Xk&v)
        Xkv, Xq = X[:,:rseq,:], X[:,rseq:,:]
        # print('###Xq',Xq.shape)
        # print('###Xkv',Xkv.shape)

        Z = []
        # Compute attention for all heads
        for i in range(self.n_heads):
            # Get Queries, Keys, Values
            Q = self.all_w[i][0](Xq)
            K = self.all_w[i][1](Xkv)
            V = self.all_w[i][2](Xkv)

            # Get attention
            Z += [self.attention(Q, K.permute(0,2,1), V)]
            
        # Concate and get the final projected Z
        Z = torch.cat(Z, dim=2)
        # print('###Z',Z.shape)
        Z = self.wo(Z)
        # print('###Z',Z.shape)

        # skip connection
        Z = Z + Xq

        # normalization
        Z = self.norm2(Z)

        # Last feed forward
        Z = self.feed_forward(Z)

        return Z



    def attention(self, Q, K, V):
        M = torch.matmul(Q,K)/(self.d_model**0.5)
        A = nn.functional.softmax(M, dim = -1)
        return torch.matmul(A,V)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class Trans2D(nn.Module):
    def __init__(self, d, n_heads = 1):
        super(Trans2D, self).__init__()

        if d%n_heads != 0:
            print("!!! d has to be a multiple of n_heads !!!")
            exit(0)
        
        
        self.d = d
        self.n_heads = n_heads
        self.d_heads = int(self.d/self.n_heads)
        
        self.pe = None 

        self.wq = nn.Linear(self.d, self.d)
        self.wk = nn.Linear(self.d, self.d)
        self.wv = nn.Linear(self.d, self.d)

        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
        
        
    def forward(self, x):
        bs, d, h, w = x.shape
        self.n = w*h
        self.w = w
        self.h = h

        if self.pe == None:
            self.pe = positionalencoding2d(self.d, self.h, self.w).cuda()


        # Positionnal encodding
        x = x + self.pe.repeat(bs,1,1,1)


        # Get queries, keys, values
        x = torch.reshape(x, (bs, self.d, self.n))
        x = x.permute(0,2,1)
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # Reshaping
        Q = torch.reshape(Q, (bs,self.n,self.n_heads, self.d_heads))
        K = torch.reshape(K, (bs,self.n,self.n_heads, self.d_heads))
        V = torch.reshape(V, (bs,self.n,self.n_heads, self.d_heads))

        # Permuting
        Q = Q.permute((0,2,1,3))
        K = K.permute((0,2,3,1))
        V = V.permute((0,2,1,3))

        # Self attention
        Y = self.attention(Q,K,V)

        # Inverse permute reshape
        Y = Y.permute(0,2,1,3)
        Y = torch.reshape(Y, (bs, self.n, self.d))
        Y = Y.permute(0,2,1)
        Y = torch.reshape(Y, (bs, self.d, self. h, self.w))
        
        return Y, Q, K, V


    def attention(self, Q, K, V):
        M = torch.matmul(Q,K)/(self.d**0.5)
        A = nn.functional.softmax(M, dim = -1)
        return torch.matmul(A,V)




class MHCA(nn.Module):
    def __init__(self, d):
        self.s_pe = None
        self.y_pe = None

        self.d = d

        self.wq = nn.Linear(2*self.d, 2*self.d)
        self.wk = nn.Linear(2*self.d, 2*self.d)
        self.wv = nn.Linear(self.d, self.d)

        self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                            nn.Conv3d(2*self.d, 2*self.d, 3, 1, 1),)

        self.conv1 = nn.Sequential(nn.Conv3d(2*self.d, 2*self.d, 1, 1, 1),
                                   nn.BatchNorm3d(2*self.d),
                                   nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv3d(self.d, self.d, 1, 1, 1),
                                   nn.BatchNorm3d(self.d),
                                   nn.ReLU(inplace=True),)
        self.conv3 = nn.Sequential(nn.Conv3d(2*self.d, self.d, 1, 1, 1),
                                   nn.BatchNorm3d(self.d),
                                   nn.ReLU(inplace=True),)

        self.sigConv = nn.Sequential(nn.Conv3d(self.d, self.d, 1, 1, 1),
                                   nn.BatchNorm3d(2*self.d),
                                   nn.Sigmoid(),
                                   nn.UpsamplingBilinear2d(scale_factor=2),)

    def forward(self, y, s):
        bs, dy, hy, wy = y.shape
        self.ny = wy*hy
        self.wy = wy
        self.hy = hy

        _, ds, hs, ws = s.shape
        self.ns = ws*hs
        self.ws = ws
        self.hs = hs

        if self.pe == None:
            self.y_pe = positionalencoding2d(self.yd, self.yh, self.yw).cuda()
            self.s_pe = positionalencoding2d(self.sd, self.sh, self.sw).cuda()


        # Positionnal encodding
        y = y + self.y_pe.repeat(bs,1,1,1)
        s = s + self.s_pe.repeat(bs,1,1,1)

        # Convs and up
        y_c1 = self.conv1(y)
        s_c2 = self.conv2(s)
        y_c3 = self.up(y)
        y_c3 = self.conv3(y_c3)


        # Get queries, keys, values
        y_c1 = torch.reshape(y_c1, (bs, 2*self.d, self.ny))
        y_c1 = y_c1.permute(0,2,1)
        s_c2 = torch.reshape(s_c2, (bs, self.d, self.ns))
        s_c2 = s_c2.permute(0,2,1)

        Q = self.wq(y_c1)
        K = self.wk(y_c1)
        V = self.wv(s_c2)

        # Reshaping
        Q = torch.reshape(Q, (bs,self.ny,self.n_heads, self.d_heads))
        K = torch.reshape(K, (bs,self.ny,self.n_heads, self.d_heads))
        V = torch.reshape(V, (bs,self.ns,self.n_heads, self.d_heads))

        # Permuting
        Q = Q.permute((0,2,1,3))
        K = K.permute((0,2,3,1))
        V = V.permute((0,2,1,3))

        # Self attention
        Z = self.attention(Q,K,V)

        # Inverse permute reshape
        Z = Z.permute(0,2,1,3)
        Z = torch.reshape(Z, (bs, self.n, self.d))
        Z = Z.permute(0,2,1)
        Z = torch.reshape(Z, (bs, self.d, self. h, self.w))
        
        # sigmoid module and up
        Z = self.sigConv(Z)
        Z = Z*s

        Z = torch.cat([Z, y_c3], 1)

        return Z, Q, K, V

    def attention(self, Q, K, V):
        M = torch.matmul(Q,K)/(self.d**0.5)
        A = nn.functional.softmax(M, dim = -1)
        return torch.matmul(A,V)