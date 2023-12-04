#usr/bin/python3

from .network_modules.unets import *
from .network_modules.attentions import *
from .network_modules.base_net import *
from .network_modules.convs import *
import torch.nn as nn
import yaml


class AifBlock(TimeConditionEmbModel):
    
    def __init__(self,dim_in,
                 dim_out,
                 dim_encoded_time,
                 num_heads=None,
                 dim_heads=None,
                 dim_condition=3,
                 size_change=False,
                 down=True,
                 linear_attention=False,
                 self_attention=False,
                 cross_attention=False,
                 ):
        super().__init__()
        self.net=EmbedSequential()
        if down:
            self.net.append(
                self.build_conv(dim_in=dim_in,dim_out=dim_out,dim_encoded_time=dim_encoded_time)
            )
            if self_attention:
                self.net.append(
                    SelfAttentionBlock(dim_in=dim_out, dim_out=dim_out, num_heads=num_heads, dim_heads=dim_heads, linear_attention=linear_attention)
                    )
            if cross_attention:
                self.net.append(
                    CrossAttentionBlock(dim_in=dim_out, dim_out=dim_out, num_heads=num_heads, dim_heads=dim_heads, linear_attention=linear_attention,dim_condition=dim_condition)
                    )
            if size_change:
                self.net.append(
                    self.build_downsample(dim=dim_out)
                )
        else:
            if size_change:
                self.net.append(
                    self.build_upsample(dim=dim_in)
                )
            if cross_attention:
                self.net.append(
                    CrossAttentionBlock(dim_in=dim_in, dim_out=dim_in, num_heads=num_heads, dim_heads=dim_heads, linear_attention=linear_attention,dim_condition=dim_condition)
                    )
            if self_attention:
                self.net.append(
                    SelfAttentionBlock(dim_in=dim_in, dim_out=dim_in, num_heads=num_heads, dim_heads=dim_heads, linear_attention=linear_attention)
                )
            self.net.append(
                    self.build_conv(dim_in=dim_in,dim_out=dim_out,dim_encoded_time=dim_encoded_time)
                )

    def forward(self, x, t,condition):
        return self.net(x,t,condition)
    
    def build_conv(self,dim_in,dim_out,dim_encoded_time):
        return DSC_Res_block(dim_in=dim_in,dim_out=dim_out,dim_encoded_time=dim_encoded_time)  
    
    def build_upsample(self,dim):
        return interpolate_up_sampling(dim=dim)
        
    def build_downsample(self,dim):
        return SPD_Conv_down_sampling(dim=dim)

class AifNet(UNet):

    def __init__(self, path_config_file:str="",**kwargs):
        self.configs_dict={"dim_basic":32, 
                    "dim_multipliers":[1,2,4,4,8,8], 
                    "attention_layers":[3,4], 
                    "condition_layers":[-2], 
                    "use_input_condition":True,
                    "skip_connection_scale":0.707, 
                    "depth_each_layer":2, 
                    "dim_encoded_time":8, 
                    "dim_in":3, 
                    "dim_out":3, 
                    "dim_condition":3, 
                    "heads_attention":4, 
                    "linear_attention":False
                    }
        if path_config_file != "":
            with open(path_config_file,"r") as f:
                yaml_configs=yaml.safe_load(f)
            for key in yaml_configs.keys():
                self.configs_dict[key]=yaml_configs[key]
        # read configs from kwargs
        for key in kwargs.keys():
            self.configs_dict[key]=kwargs[key]
        for key in self.configs_dict.keys():
            setattr(self,key,self.configs_dict[key])    
    
        if self.use_input_condition:
            self.dim_in=self.dim_in+self.dim_condition
        super().__init__(self.dim_in, self.dim_out, self.dim_basic, self.dim_multipliers, self.skip_connection_scale)
        self.time_encoding = TimeEncoding(self.dim_encoded_time)

    def build_initial_block(self, dim_in: int, dim_out: int):
        return nn.Conv2d(dim_in, dim_out, 1, padding=0)

    def build_down_block(self, dim_in: int, dim_out: int, index_layer: int):
        self_attention = False
        condition_attention = False
        down = nn.Sequential()
        if index_layer in self.attention_layers:
            self_attention = True
        if index_layer in self.condition_layers:
            condition_attention = True
        if self.depth_each_layer == 1:
            # dim change + attention + change_size
            down.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=True)
                        )
        elif self.depth_each_layer ==2:
            # dim change
            down.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=True)
                        )
            # attention + change_size
            down.append(self.build_Aif_Block(dim_in=dim_out,dim_out=dim_out,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=True)
                        )
        else:
            # dim change
            down.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=True)
                        )
            # pure conve
            for i in range(self.depth_each_layer-2):
                down.append(self.build_Aif_Block(dim_in=dim_out,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=True)
                            )
            # attention + change_size
            down.append(self.build_Aif_Block(dim_in=dim_out,dim_out=dim_out,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=True)
                        )            
        return down

    def build_up_block(self, dim_in: int, dim_out: int, index_layer: int):
        self_attention = False
        condition_attention = False
        up = nn.Sequential()
        if index_layer in self.attention_layers:
            self_attention = True
        if index_layer in self.condition_layers:
            condition_attention = True
        if self.depth_each_layer == 1:
            # dim change + attention + change_size
            up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=False)
                        )
        elif self.depth_each_layer ==2:
            # attention + change_size
            up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_in//2,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=False)
                        )
            # dim change
            up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=False)
                        )
        else:
            # attention + change_size 
            up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_in//2,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=False)
                        )
            # pure conve
            for i in range(self.depth_each_layer-2):
                up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_in//2,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=False)
                            )
            # dim change
            up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=False)
                        )         
        return up

    def build_bottleneck(self, dim_in: int, dim_out: int):
        bottle = EmbedSequential()
        bottle.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=True,condition_attention=False,
                                             size_change=False,
                                             down=True)
                      )
        bottle.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=False)
                      )
        return bottle

    def build_final_block(self, dim_in: int, dim_out: int):
        final = EmbedSequential()
        final.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=False,condition_attention=False,
                                             size_change=False,
                                             down=False)
                     )
        final.append(nn.Conv2d(dim_out, dim_out, 1))
        return final

    def build_Aif_Block(self,dim_in,dim_out,down,self_attention, condition_attention,size_change):
        return AifBlock(dim_in=dim_in, dim_out=dim_out, dim_encoded_time=self.dim_encoded_time,
                                self_attention=self_attention, cross_attention=condition_attention,
                                size_change=size_change,
                                down=down,
                                num_heads=self.heads_attention, dim_heads=dim_out // self.heads_attention, 
                                linear_attention=self.linear_attention,
                                dim_condition=self.dim_condition)

    def forward(self, x, t, condition):
        if self.use_input_condition:
            x=torch.cat((x,condition),1)
        t = self.time_encoding(t)
        skips = []
        x = self.initial_layer(x)
        for down_block in self.down_nets:
            for d_net in down_block:
                x = d_net(x, t, condition)
                skips.append(x*self.skip_connection_scale)
        x = self.bottle_neck(x, t, condition)
        for up_block in self.up_nets:
            for u_net in up_block:
                x = torch.cat((x, skips.pop()), dim=1)
                x = u_net(x, t, condition)
        x = self.final_layer(x, t, condition)
        return x