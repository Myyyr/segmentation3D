import torch
import torch.nn as nn
import math
from models.networks_other import init_weights

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class MHCrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(MHCrossTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MHCrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, rseq=512):
        for attn, ff in self.layers:
            x = attn(x, rseq) + x
            x = ff(x) + x
        return x

class MHCrossAttention(nn.Module):
    """docstring for MHCrossAttention"""

    # !! Maybe add dropout later !!
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(MHCrossAttention, self).__init__()
        self.dim = dim
        self.inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.wq = nn.Linear(self.dim, self.inner_dim, bias = False)
        self.wk = nn.Linear(self.dim, self.inner_dim, bias = False)
        self.wv = nn.Linear(self.dim, self.inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        #initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')        


    def forward(self, X, rseq=512):
        # Normalization
        # print('###X',X.shape)
        X = self.norm1(X)
        # print('###X',X.shape)

        # Separation Region/FullImage into Xq / (Xk&v)
        Xq, Xkv = X[:,:rseq,:], X[:,rseq:,:]
        # print('###Xq',Xq.shape)
        # print('###Xkv',Xkv.shape)

        Z = []
        # Compute attention for all heads
        Q = self.wq(Xq)
        K = self.wk(Xk)
        V = self.wv(XV)

        Q = rearrange(Q, 'b n (h d) -> b n h d',h=self.heads)
        K = rearrange(K, 'b n (h d) -> b n h d',h=self.heads)
        V = rearrange(V, 'b n (h d) -> b n h d',h=self.heads)

        Z = self.attention(Q,K,V)
        del Q, K, V
        Z = rearrange(Z, 'b h n d -> b n (h d)')
        Z = self.to_out(Z)

        return Z



    def attention(self, Q, K, V):
        M = torch.matmul(Q,K.transpose(-2,-1))*self.scale
        A = nn.functional.softmax(M, dim = -1)
        del M
        return torch.matmul(A,V)

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
            wq = nn.Linear(self.d_model, self.d_model, bias=False)
            wk = nn.Linear(self.d_model, self.d_model, bias=False)
            wv = nn.Linear(self.d_model, self.d_model, bias=False)
            self.all_w.append(nn.ModuleList([wq, wk, wv]))
        self.all_w = nn.ModuleList(self.all_w)

        self.wo = nn.Linear(self.d_model*self.n_heads, self.d_model, bias=False)
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
        Xq, Xkv = X[:,:rseq,:], X[:,rseq:,:]
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
            # Z += [self.attention(Q, K.permute(0,2,1), V)]
            Z += [self.attention(Q, K, V)]

        # Concate and get the final projected Z
        Z = torch.cat(Z, dim=2)
        # print('###Z',Z.shape)
        Z = self.wo(Z)
        # print('###Z',Z.shape)

        # skip connection
        Z1 = Z + Xq
        del X, Xq, Xkv

        # normalization
        Z = self.norm2(Z1)

        # Last feed forward
        Z = self.feed_forward(Z) + Z1
        del Z1

        return Z



    def attention(self, Q, K, V):
        # print(Q.shape,K.shape,V.shape)
        # print(torch.cuda.memory_allocated()*4/(1024**3), 'GB')
        # exit(0)
        M = torch.matmul(Q,K.transpose(-2,-1))/(self.d_model**0.5)
        # M = torch.einsum('b i d, b j d -> b i j', Q, K)/(self.d_model**0.5)
        # print(M.shape)
        A = nn.functional.softmax(M, dim = -1)
        del M
        # print("We do it goooood !")
        # return torch.einsum('b i j, b j d -> b i d', A, V)
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





