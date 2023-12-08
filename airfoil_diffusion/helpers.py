# helpers has been released separately as foxutils: https://github.com/qiauil/foxutils

import yaml
import numpy as np
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def show_paras(model,print_result=True):
    nn_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in nn_parameters])
    # crucial parameter to keep in view: how many parameters do we have?
    if print_result:
        print("model has {} trainable params".format(params))
    return params

class GeneralDataClass():
    
    def __init__(self,generation_dict=None,**kwargs) -> None:
        if generation_dict is not None:
            for key,value in generation_dict.items():
                self.set(key,value)
        for key,value in kwargs.items():
            self.set(key,value)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self,key):
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__.items())
    
    def keys(self):
        return self.__dict__.keys()

    def set(self,key,value):
        if isinstance(value,dict):
            setattr(self,key,GeneralDataClass(value))
        else:
            setattr(self,key,value)

    def set_items(self,**kwargs):
        for key,value in kwargs.items():
            self.set(key,value)
    
    def remove(self,*args):
        for key in args:
            delattr(self,key)

class ConfigurationsHandler():
    
    def __init__(self) -> None:
        self.__configs_feature={}
        self.__configs=GeneralDataClass()
        
    def add_config_item(self,name,default_value=None,default_value_func=None,mandatory=False,description="",value_type=None,option=None,in_func=None,out_func=None):
        if not mandatory and default_value is None and default_value_func is None:
            raise Exception("Default value or default value func must be set for non-mandatory configuration.")
        if mandatory and (default_value is not None or default_value_func is not None):
            raise Exception("Default value or default value func must not be set for mandatory configuration.")
        if default_value is not None and type(default_value)!=value_type:
            raise Exception("Default value must be {}, but find {}.".format(value_type,type(default_value)))
        if option is not None:
            if type(option)!=list:
                raise Exception("Option must be list, but find {}.".format(type(option)))
            if len(option)==0:
                raise Exception("Option must not be empty.")
            for item in option:
                if type(item)!=value_type:
                    raise Exception("Option must be list of {}, but find {}.".format(value_type,type(item)))
        self.__configs_feature[name]={
            "default_value_func":default_value_func, #default_value_func must be a function with one parameter, which is the current configures
            "mandatory":mandatory,
            "description":description,
            "value_type":value_type,
            "option":option,
            "in_func":in_func,
            "out_func":out_func,
            "default_value":default_value,
            "in_func_ran":False,
            "out_func_ran":False
        }
    
    def get_config_features(self,key):
        if key not in self.__configs_feature.keys():
            raise Exception("{} is not a supported configuration.".format(key))
        return self.__configs_feature[key]
    
    def set_config_features(self,key,feature):
        self.add_config_item(key,default_value=feature["default_value"],
                             default_value_func=feature["default_value_func"],
                             mandatory=feature["mandatory"],
                             description=feature["description"],
                             value_type=feature["value_type"],
                             option=feature["option"],
                             in_func=feature["in_func"],
                             out_func=feature["out_func"])

    def set_config_items(self,**kwargs):
        for key in kwargs.keys():
            if key not in self.__configs_feature.keys():
                raise Exception("{} is not a supported configuration.".format(key))
            if self.__configs_feature[key]["value_type"] is not None and type(kwargs[key])!=self.__configs_feature[key]["value_type"]:
                raise Exception("{} must be {}, but find {}.".format(key,self.__configs_feature[key]["value_type"],type(kwargs[key])))
            if self.__configs_feature[key]["option"] is not None and kwargs[key] not in self.__configs_feature[key]["option"]:
                raise Exception("{} must be one of {}, but find {}.".format(key,self.__configs_feature[key]["option"],kwargs[key]))
            self.__configs.set(key,kwargs[key])
            self.__configs_feature[key]["in_func_ran"]=False
            self.__configs_feature[key]["out_func_ran"]=False
    
    def configs(self):
        for key in self.__configs_feature.keys():
            not_set=False
            if not hasattr(self.__configs,key):
                not_set=True
            elif self.__configs[key] is None:
                not_set=True
            if not_set:
                if self.__configs_feature[key]["mandatory"]:
                    raise Exception("Configuration {} is mandatory, but not set.".format(key))
                elif self.__configs_feature[key]["default_value"] is not None:
                    self.__configs.set(key,self.__configs_feature[key]["default_value"])
                    self.__configs_feature[key]["in_func_ran"]=False
                    self.__configs_feature[key]["out_func_ran"]=False
                elif self.__configs_feature[key]["default_value_func"] is not None:
                    self.__configs.set(key,None)        
                else:
                    raise Exception("Configuration {} is not set.".format(key))
        #default_value_func and infunc may depends on other configurations
        for key in self.__configs.keys():
            if self.__configs[key] is None and self.__configs_feature[key]["default_value_func"] is not None:
                self.__configs.set(key,self.__configs_feature[key]["default_value_func"](self.__configs))
                self.__configs_feature[key]["in_func_ran"]=False
                self.__configs_feature[key]["out_func_ran"]=False
        for key in self.__configs_feature.keys():
            if self.__configs_feature[key]["in_func"] is not None and not self.__configs_feature[key]["in_func_ran"]:
                self.__configs.set(key,self.__configs_feature[key]["in_func"](self.__configs[key],self.__configs))
                self.__configs_feature[key]["in_func_ran"]=True
        return self.__configs

    def set_config_items_from_yaml(self,yaml_file):
        with open(yaml_file,"r") as f:
            yaml_configs=yaml.safe_load(f)
        self.set_config_items(**yaml_configs)
    
    def save_config_items_to_yaml(self,yaml_file,only_optional=False):
        config_dict=self.configs().__dict__
        if only_optional:
            output_dict={}
            for key in config_dict.keys():
                if self.__configs_feature[key]["mandatory"]:
                    continue
                output_dict[key]=config_dict[key]    
        else:
            output_dict=config_dict
        for key in output_dict.keys():
            if self.__configs_feature[key]["out_func"] is not None and not self.__configs_feature[key]["out_func_ran"]:
                output_dict[key]=self.__configs_feature[key]["out_func"](self.__configs[key],self.__configs)
                self.__configs_feature[key]["out_func_ran"]=True
        with open(yaml_file,"w") as f:
            yaml.dump(output_dict,f)
    
    def show_config_features(self):
        mandatory_configs=[]
        optional_configs=[]
        for key in self.__configs_feature.keys():
            text="    "+str(key)
            texts=[]
            if self.__configs_feature[key]["value_type"] is not None:
                texts.append(str(self.__configs_feature[key]["value_type"].__name__))
            if self.__configs_feature[key]["option"] is not None:
                texts.append("possible option: "+str(self.__configs_feature[key]["option"]))
            if self.__configs_feature[key]["default_value"] is not None:
                texts.append("default value: "+str(self.__configs_feature[key]["default_value"]))
            if len(texts)>0:
                text+=" ("+", ".join(texts)+")"
            text+=": "
            text+=str(self.__configs_feature[key]["description"])
            if self.__configs_feature[key]["mandatory"]:
                mandatory_configs.append(text)
            else:
                optional_configs.append(text)
        print("Mandatory Configuration:")
        for key in mandatory_configs:
            print(key)
        print("")
        print("Optional Configuration:")
        for key in optional_configs:
            print(key)
   
    def show_config_items(self):
        for key,value in self.configs().__dict__.items():
            print("{}: {}".format(key,value))
