#usr/bin/python3
import torch.nn as nn
import math,torch
from ..helpers import *

class UNet(nn.Module):
    def __init__(
        self,
        path_config_file:str="",**kwargs
    ):
        super().__init__()
        if not hasattr(self,"configs_handler"):
            self.configs_handler=ConfigurationsHandler()
        self.configs_handler.add_config_item("dim_in",mandatory=True,value_type=int,description="The input dim of the model.")
        self.configs_handler.add_config_item("dim_out",mandatory=True,value_type=int,description="The output dim of the model.")
        self.configs_handler.add_config_item("dim_basic",mandatory=True,value_type=int,description=r"The basic dimensions in each layer. The real dimension numbers are dim_basic$\times$dim_multipliers.")
        self.configs_handler.add_config_item("condition_dim",mandatory=False,default_value=0,value_type=int,description="The dimensions of conditions. Please set this value as 0 when no condition is provided.")
        self.configs_handler.add_config_item("dim_multipliers",mandatory=True,value_type=list,description=r"A list used to control the depth and the size of the net. There will be len(dim_multipliers)-1 down/up blocks in the net. The number of input/output channels of each block will also be determined by the elements of this list(dim_basic$\times$dim_multipliers). For instance, if the dim_multipliers is [1 2 4 8]. There will be 3 down/up blocks. The input/output channel of these blocks are (dim_basic, 2$\times$dim_basic), (2$\times$dim_basic, 4$\times$dim_basic) and (4$\times$dim_basic, 8$\times$dim_basic). The size of neckblock will be  8$\times$dim_basic $\times$ input_channel/$2^3$ $\times$ input_channel/$2^3$. If the first elements is 0, the input channel of the first down layer will be the dim_in and the output channel of the last down layer will be dim_out.")
        self.configs_handler.add_config_item("skip_connection_scale",default_value=1.0,value_type=float,description="The scale of the skip connection. The output of each down block will be multiplied by this value before being added to the input of the corresponding up block.")
        if path_config_file!="":
            self.configs_handler.set_config_items_from_yaml(path_config_file)
        self.configs_handler.set_config_items(**kwargs)
        self.configs=self.configs_handler.configs()
        

        self.channels = [self.configs.dim_basic*i for i in self.configs.dim_multipliers]
        self.in_out_pairs = list(zip(self.channels[:-1], self.channels[1:]))
        self.skip_connection_scale=self.configs.skip_connection_scale

        self.initial_layer=self.build_initial_block(self.configs.dim_in,self.channels[0])
        self.down_nets = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(self.in_out_pairs):
            if dim_in == 0:
               dim_in = self.configs.dim_in
            self.down_nets.append(self.build_down_block(dim_in=dim_in,dim_out=dim_out,index_layer=i+1))
        self.bottle_neck = self.build_bottleneck(dim_in=self.channels[-1],dim_out=self.channels[-1])
        self.up_nets = nn.ModuleList([])
        for i, (dim_out, dim_in) in enumerate(reversed(self.in_out_pairs)):
            dim_in = 2*dim_in
            if dim_out == 0:
               dim_out = self.configs.dim_out
            self.up_nets.append(self.build_up_block(dim_in=dim_in,dim_out=dim_out,index_layer=len(self.channels)-1-i))
        self.final_layer = self.build_final_block(dim_in=self.channels[0],dim_out=self.configs.dim_out)
    
    def build_initial_block(self,dim_in:int,dim_out:int):
        return nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def build_down_block(self,dim_in:int,dim_out:int,index_layer:int):
        return nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1)

    def build_bottleneck(self,dim_in:int,dim_out:int):
        return nn.Conv2d(dim_in, dim_out, 3, padding=1)
    
    def build_up_block(self,dim_in:int,dim_out:int,index_layer:int):
        return nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1)

    def build_final_block(self,dim_in:int,dim_out:int):
        return nn.Conv2d(dim_in, dim_out, 3, padding=1)

    def check_size(self, input_size: int):
        size_neck = input_size/(math.pow(2, len(self.in_out_pairs)))
        print("The image size of the neck layer is (B,{},{},{})".format(
            self.in_out_pairs[-1][-1], size_neck, size_neck))

    def save_configs(self,yaml_file):
        self.configs_handler.save_config_items_to_yaml(yaml_file)
    
    def show_config_options(self):
        self.configs_handler.show_config_features()
    
    def show_current_configs(self):
        self.configs_handler.show_config_items()
        
    def save_current_configs(self,yaml_file):
        self.configs_handler.save_config_items_to_yaml(yaml_file,only_optional=True)

    def forward(self, x):
        skips = []
        x = self.initial_layer(x)
        for down_block in self.down_nets:
            x = down_block(x)
            skips.append(x*self.skip_connection_scale)
        x = self.bottle_neck(x)
        for up_block in self.up_nets:
            x = torch.cat((x, skips.pop()), dim=1)
            x = up_block(x)
        x = self.final_layer(x)
        return x
