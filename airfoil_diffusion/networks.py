#usr/bin/python3

from .network_modules.unets import *
from .network_modules.attentions import *
from .network_modules.base_net import *
from .network_modules.convs import *
from .helpers import *
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
        if not hasattr(self,"configs_handler"):
            self.configs_handler=ConfigurationsHandler()
        self.configs_handler.add_config_item("dim_in",mandatory=False,default_value=3,value_type=int,description="The input dim of the model.")
        self.configs_handler.add_config_item("dim_out",mandatory=False,default_value=3,value_type=int,description="The output dim of the model.")
        self.configs_handler.add_config_item("dim_basic",mandatory=False,default_value=32,value_type=int,description=r"The basic dimensions in each layer. The real dimension numbers are dim_basic$\times$dim_multipliers.")
        self.configs_handler.add_config_item("dim_multipliers",mandatory=False,default_value=[1,2,4,4,8,8],value_type=list,description=r"A list used to control the depth and the size of the net. There will be len(dim_multipliers)-1 down/up blocks in the net. The number of input/output channels of each block will also be determined by the elements of this list(dim_basic$\times$dim_multipliers). For instance, if the dim_multipliers is [1 2 4 8]. There will be 3 down/up blocks. The input/output channel of these blocks are (dim_basic, 2$\times$dim_basic), (2$\times$dim_basic, 4$\times$dim_basic) and (4$\times$dim_basic, 8$\times$dim_basic). The size of neckblock will be  8$\times$dim_basic $\times$ input_channel/$2^3$ $\times$ input_channel/$2^3$. If the first elements is 0, the input channel of the first down layer will be the dim_in and the output channel of the last down layer will be dim_out.")
        self.configs_handler.add_config_item("attention_layers",mandatory=False,default_value=[3,4],value_type=list,description="The layers where attention blocks are added.")
        self.configs_handler.add_config_item("condition_layers",mandatory=False,default_value=[-2],value_type=list,description="The layers where condition are added using cross attention. '-2' means that we won't use cross attention to add the condition.")
        self.configs_handler.add_config_item("use_input_condition",mandatory=False,default_value=True,value_type=bool,description="Whether to add the condition into input channels;.")
        self.configs_handler.add_config_item("skip_connection_scale",mandatory=False,default_value=0.707,value_type=float,description="The scale factor of the skip connection.")
        self.configs_handler.add_config_item("depth_each_layer",mandatory=False,default_value=2,value_type=int,description="The depth of each layer.")
        self.configs_handler.add_config_item("dim_encoded_time",mandatory=False,default_value=8,value_type=int,description="The dimension of the time embeddings.")
        self.configs_handler.add_config_item("dim_condition",mandatory=False,default_value=3,value_type=int,description="The dimension of the condition.")
        self.configs_handler.add_config_item("heads_attention",mandatory=False,default_value=4,value_type=int,description="The number of heads in the attention blocks.") 
        self.configs_handler.add_config_item("linear_attention",mandatory=False,default_value=False,value_type=bool,description="Whether to use linear attention.")       
        if path_config_file != "":
            self.configs_handler.set_config_items_from_yaml(path_config_file)
        self.configs_handler.set_config_items(**kwargs)
        self.configs=self.configs_handler.configs() 
    
        if self.configs.use_input_condition:
            self.configs.dim_in=self.configs.dim_in+self.configs.dim_condition
        super().__init__(dim_in=self.configs.dim_in, 
                         dim_out=self.configs.dim_out, 
                         dim_basic=self.configs.dim_basic, 
                         dim_multipliers=self.configs.dim_multipliers, 
                         skip_connection_scale=self.configs.skip_connection_scale)
        self.time_encoding = TimeEncoding(self.configs.dim_encoded_time)

    def build_initial_block(self, dim_in: int, dim_out: int):
        return nn.Conv2d(dim_in, dim_out, 1, padding=0)

    def build_down_block(self, dim_in: int, dim_out: int, index_layer: int):
        self_attention = False
        condition_attention = False
        down = nn.Sequential()
        if index_layer in self.configs.attention_layers:
            self_attention = True
        if index_layer in self.configs.condition_layers:
            condition_attention = True
        if self.configs.depth_each_layer == 1:
            # dim change + attention + change_size
            down.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=True)
                        )
        elif self.configs.depth_each_layer ==2:
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
            for i in range(self.configs.depth_each_layer-2):
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
        if index_layer in self.configs.attention_layers:
            self_attention = True
        if index_layer in self.configs.condition_layers:
            condition_attention = True
        if self.configs.depth_each_layer == 1:
            # dim change + attention + change_size
            up.append(self.build_Aif_Block(dim_in=dim_in,dim_out=dim_out,
                                             self_attention=self_attention,condition_attention=condition_attention,
                                             size_change=True,
                                             down=False)
                        )
        elif self.configs.depth_each_layer ==2:
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
            for i in range(self.configs.depth_each_layer-2):
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
        return AifBlock(dim_in=dim_in, dim_out=dim_out, dim_encoded_time=self.configs.dim_encoded_time,
                                self_attention=self_attention, cross_attention=condition_attention,
                                size_change=size_change,
                                down=down,
                                num_heads=self.configs.heads_attention, dim_heads=dim_out // self.configs.heads_attention, 
                                linear_attention=self.configs.linear_attention,
                                dim_condition=self.configs.dim_condition)

    def forward(self, x, t, condition):
        if self.configs.use_input_condition:
            x=torch.cat((x,condition),1)
        t = self.time_encoding(t)
        skips = []
        x = self.initial_layer(x)
        for down_block in self.down_nets:
            for d_net in down_block:
                x = d_net(x, t, condition)
                skips.append(x*self.configs.skip_connection_scale)
        x = self.bottle_neck(x, t, condition)
        for up_block in self.up_nets:
            for u_net in up_block:
                x = torch.cat((x, skips.pop()), dim=1)
                x = u_net(x, t, condition)
        x = self.final_layer(x, t, condition)
        return x