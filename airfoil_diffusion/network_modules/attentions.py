#usr/bin/python3
import torch,math
from torch import nn
from einops import rearrange
from .base_net import *
from ..helpers import *


class ScaledDotProductAttention(nn.Module):
        
    def __init__(self,dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d_k = keys.shape[-1]
        weights = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d_k)
        weights = nn.functional.softmax(weights, dim=-1)
        return torch.bmm(self.dropout(weights), values)

class LinearScaledDotProductAttention(nn.Module):
        
    def __init__(self,dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        weights=(nn.functional.softmax(keys,dim=-2)).transpose(1,2)
        weights=torch.bmm(weights,values)
        return torch.bmm(nn.functional.softmax(queries,dim=-1),self.dropout(weights))

class MultiHeadAttentionBase(nn.Module):
    
    def __init__(self, num_heads:int,linear_attention=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        if linear_attention:
            self.attention=LinearScaledDotProductAttention(dropout=dropout)
        else:
            self.attention = ScaledDotProductAttention(dropout=dropout)


    def forward(self, queries, keys, values):
        queries,keys,values =map(self.apart_input,(queries,keys,values))
        output = self.attention(queries, keys, values)
        return self.concat_output(output)


    def apart_input(self,x):
        #(batch_size, num_elements, num_heads$\times$dim_deads)  >>> (batch_size, num_elements, num_heads, dim_deads)
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        #(batch_size, num_elements, num_heads, dim_deads) >>> (batch_size, num_heads, num_elements, dim_deads) 
        x = x.permute(0, 2, 1, 3)
        #(batch_size, num_heads, num_elements, dim_deads)  >>> (batch_size$\times$num_heads, num_elements, dim_deads) 
        return x.reshape(-1, x.shape[2], x.shape[3])


    def concat_output(self, x):
        #(batch_size$\times$num_heads, num_elements, dim_deads) >>> (batch_size, num_heads, num_elements, dim_deads)
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        #(batch_size, num_heads, num_elements, dim_deads) >>> (batch_size, num_elements, num_heads, dim_deads)
        x = x.permute(0, 2, 1, 3)
        #(batch_size, num_elements, num_heads, dim_deads) >>> (batch_size, num_elements, num_heads$\times$dim_deads)
        return x.reshape(x.shape[0], x.shape[1], -1)

class SequenceMultiHeadAttention(nn.Module):
    def __init__(self,dim_q:int, dim_k:int, dim_v:int, num_heads:int, dim_heads:int,dim_out:int, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Linear(dim_q, dim_hiddens,bias=bias)
        self.w_k = nn.Linear(dim_k, dim_hiddens,bias=bias)
        self.w_v = nn.Linear(dim_v, dim_hiddens,bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Linear(dim_hiddens, dim_out,bias=bias)
    
    def forward(self, queries, keys, values):
        q=self.w_q(queries)
        k=self.w_k(keys)
        v=self.w_v(values)
        att=self.mha(q,k,v)
        return self.w_o(att)

class TwoDFieldMultiHeadAttention(nn.Module):

    def __init__(self,dim_q, dim_k, dim_v, num_heads, dim_heads,dim_out, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Conv2d(dim_q, dim_hiddens, 1, bias=bias)
        self.w_k = nn.Conv2d(dim_k, dim_hiddens, 1, bias=bias)
        self.w_v = nn.Conv2d(dim_v, dim_hiddens, 1, bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Conv2d(dim_hiddens, dim_out,1,bias=bias)
    
    def forward(self, queries, keys, values):
        width=queries.shape[-1]
        q=self.w_q(queries)
        k=self.w_k(keys)
        v=self.w_v(values)
        q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), (q,k,v))
        att=self.mha(q,k,v)
        att_2D=rearrange(att,"b (h w) c -> b c h w",w=width)
        return self.w_o(att_2D)

class TwoDFieldMultiHeadSelfAttention(nn.Module):

    def __init__(self,dim_in:int, num_heads:int, dim_heads:int,dim_out:int, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_qkv=nn.Conv2d(dim_in, dim_hiddens*3, 1, bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Conv2d(dim_hiddens, dim_out,1,bias=bias)
    
    def forward(self, x):
        width=x.shape[-1]
        qkv = self.w_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b c h w -> b (h w) c"), qkv)
        att=self.mha(q,k,v)
        att_2D=rearrange(att,"b (h w) c -> b c h w",w=width)
        return self.w_o(att_2D)

class TwoDFieldMultiHeadChannelSelfAttention(nn.Module):

    def __init__(self,num_pixel:int, num_heads:int, dim_heads:int, linear_attention=False, dropout=0.0,bias=False):
        super().__init__()
        dim_hiddens=num_heads*dim_heads
        self.w_q = nn.Linear(num_pixel, dim_hiddens,bias=bias)
        self.w_k = nn.Linear(num_pixel, dim_hiddens,bias=bias)
        self.w_v = nn.Linear(num_pixel, dim_hiddens,bias=bias)
        self.mha=MultiHeadAttentionBase(num_heads=num_heads,linear_attention=linear_attention,dropout=dropout)
        self.w_o = nn.Linear(dim_hiddens, num_pixel,bias=bias)
    
    def forward(self, x):
        h=x.shape[-1]
        x=rearrange(x, "b c h w -> b c (hw)")
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        att=self.w_o(self.mha(q,k,v))
        return rearrange(att, "b c (hw) -> b c h w",h=h)

class PositionalEncoding(nn.Module):

    def __init__(self,dim:int,max_elements_num=10000,dropout=0.0):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        indexes_row = torch.arange(max_elements_num, dtype=torch.float32).reshape(-1, 1)
        indexes_col = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        x = indexes_row / torch.pow(10000, indexes_col)
        self.position=torch.zeros((1,max_elements_num,dim))
        self.position[:, :, 0::2] = torch.sin(x)
        self.position[:, :, 1::2] = torch.cos(x)      
    
    def forward(self,x):
        x=x+self.position[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class AttentionBlockBase(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dim_heads: int, dim_condition: int, linear_attention=False, dropout=0.0):
        super().__init__()
        dim_k_v = default(dim_condition, dim_in)
        self.att = TwoDFieldMultiHeadAttention(dim_q=dim_in, dim_k=dim_k_v, dim_v=dim_k_v, num_heads=num_heads,
                                               dim_heads=dim_heads, dim_out=dim_out, linear_attention=linear_attention, dropout=dropout)
        self.residual_conv = self.res_conv = nn.Conv2d(
            dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.normal = GroupNormalX(dim_out)

    def forward(self, x, condition):
        k_v = default(condition, x)
        x = self.dropout(self.att(queries=x, keys=k_v,
                         values=k_v))+self.residual_conv(x)
        return self.normal(x)

class SelfAttentionBlock(AttentionBlockBase):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dim_heads: int, linear_attention=False, dropout=0):
        super().__init__(dim_in, dim_out, num_heads, dim_heads, dim_condition=None,
                         linear_attention=linear_attention, dropout=dropout)

    def forward(self, x):
        return super().forward(x, condition=None)

class CrossAttentionBlock(AttentionBlockBase,ConditionEmbModel):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dim_heads: int, dim_condition: int, linear_attention=False, dropout=0):
        super().__init__(dim_in, dim_out, num_heads, dim_heads,
                         dim_condition, linear_attention, dropout)

class ChannelAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_pixel: int, num_heads: int, dim_heads: int, linear_attention=False, dropout=0.0):
        super().__init__()
        self.att = TwoDFieldMultiHeadChannelSelfAttention(
            num_pixel=num_pixel, num_heads=num_heads, dim_heads=dim_heads, linear_attention=linear_attention, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.normal = GroupNormalX(dim)

    def forward(self, x):
        x = self.dropout(self.att(x))+x
        return self.normal(x)
