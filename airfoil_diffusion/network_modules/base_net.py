#usr/bin/python3
from torch import nn
import math
from abc import abstractmethod
from einops.layers.torch import Rearrange
from einops import rearrange
from .helpers import *

class TimeEncoding(nn.Module):
    def __init__(self, dim_encoding:int):
        super().__init__()
        self.dim = dim_encoding

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeEmbedding(nn.Module):

    def __init__(self,dim_input:None,dim_encoded_time=None,trainable=True):
        super().__init__()
        if trainable:
            if dim_input is None or dim_encoded_time is None:
                raise RuntimeError("'dim_input' and 'dim_encoded_time' must be specficed when trainable is True")
            else:
                self.linear1=nn.Linear(dim_encoded_time,dim_input)
                self.activation=nn.GELU()
                self.linear2=nn.Linear(dim_input,dim_input)
        else:
            self.linear1=nn.Identity()
            self.activation=nn.Identity()
            self.linear2=nn.Identity()
    
    def forward(self,x, t):
        t=self.linear2(self.activation(self.linear1(t)))
        return x + rearrange(t, "b c -> b c 1 1")

class TimeEmbModel(nn.Module):

    @abstractmethod
    def forward(self, x, t):
        """
        ...
        """

class ConditionEmbModel(nn.Module):

    @abstractmethod
    def forward(self, x, condition):
        """
        ...
        """

class TimeConditionEmbModel(nn.Module):

    @abstractmethod
    def forward(self, x, t, condition):
        """
        ...
        """

class EmbedSequential(nn.Sequential,TimeConditionEmbModel):
    def forward(self, x, t, condition):
        for layer in self:
            if isinstance(layer, TimeEmbModel) or isinstance(layer, TimeEmbedding):
                x = layer(x, t)
            elif isinstance(layer, ConditionEmbModel):
                x = layer(x, condition)
            elif isinstance(layer, TimeConditionEmbModel):
                x = layer(x,t,condition)
            else:
                x = layer(x)
        return x

class GroupNormalX(nn.Module):
    def __init__(self, dim, dim_eachgroup=16):
        super().__init__()
        groups = dim//dim_eachgroup
        if groups == 0:
            groups += 1
        self.norm = nn.GroupNorm(groups, dim)

    def forward(self, x):
        return self.norm(x)

def SPD_Conv_down_sampling(dim):
    """
    No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects
    https://arxiv.org/abs/2208.03641
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim, 1),
    )

def interpolate_up_sampling(dim):
    
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim, 3, padding=1)  
    )
