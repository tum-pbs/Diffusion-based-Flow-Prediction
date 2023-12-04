# usr/bin/python3
# trainer from foxutils: https://github.com/qiauil/foxutils
# version:0.0.4
# last modified:20231107

import os,torch,time,math,logging,yaml
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.data import DataLoader 
#from .helper.coding import *
#from .helper.network import *
import copy,random
import numpy as np
from tqdm import tqdm

# moved from foxutils.helper:
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
                if isinstance(value,dict):
                    setattr(self,key,GeneralDataClass(value))
                else:
                    setattr(self,key,value)
        for key,value in kwargs.items():
            if isinstance(value,dict):
                setattr(self,key,GeneralDataClass(value))
            else:
                setattr(self,key,value)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self,key):
        return self.__dict__[key]
    
    def __iter__(self):
        return iter(self.__dict__.items())


#NOTE: the range of the lambda output should be [0,1]                   
def get_cosine_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    def cosine_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1-(1-(math.cos((idx_epoch-warmup_epoch)/(epochs-warmup_epoch)*math.pi)+1)/2)*(1-final_lr/initial_lr)
    return cosine_lambda
    
def get_linear_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    def linear_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1-((idx_epoch-warmup_epoch)/(epochs-warmup_epoch))*(1-final_lr/initial_lr)
    return linear_lambda

def get_constant_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    def constant_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1
    return constant_lambda

class Trainer():
    
    def __init__(self) -> None:
        self.set_configs_type()
        self.__train_from_checkpoint=False
        self.configs=None
        self.project_path=""
        self.logger=None
        self.recorder=None
        self.train_dataloader=None
        self.validate_dataloader=None
        self.start_epoch=1
        self.optimizer=None
        self.lr_scheduler=None

    def __prepare_config_dict_for_inoutput(self,configs_dict,output=False):
        '''
        Preprocess the config dict before training or saving.
        
        Args:
            configs_dict: dict, the config dict to be processed
            save: bool, whether the config dict is for saving or training
        '''
        if not output:
            configs_dict["device"]=torch.device(configs_dict["device"])
        else:
            configs_dict["device"]=str(configs_dict["device"])
        return configs_dict
            
    def __check_configs_dict(self,configs_dict):
        # check whether all mandatory configs are provided
        if configs_dict["save_path"][-1]!=os.sep:
            configs_dict["save_path"]+=os.sep
        if configs_dict["batch_size_validation"] is None:
            configs_dict["batch_size_validation"]=configs_dict["batch_size_train"]
        if configs_dict["shuffle_validation"] is None:
            configs_dict["shuffle_validation"]=configs_dict["shuffle_train"]
        if configs_dict["num_workers_validation"] is None:
            configs_dict["num_workers_validation"]=configs_dict["num_workers_train"]
        if configs_dict["save_epoch"] is None:
            configs_dict["save_epoch"]=int(configs_dict["epochs"]/10)
        if configs_dict["final_lr"] is None:
            configs_dict["final_lr"]=configs_dict["lr"]
        if configs_dict["random_seed"] is None:
            configs_dict["random_seed"]=int(time.time())
        for key in self.mandatory_configs:
            if key not in configs_dict.keys():
                raise KeyError("Mandatory configuration '{}' not found in configs".format(key))
        for key in self.configs_option.keys():
            if configs_dict[key] not in self.configs_option[key]:
                raise ValueError("Config '{}' should be one of {},find {}".format(key,self.configs_option[key],configs_dict[key]))
        for key in configs_dict.keys():
            if key not in self.mandatory_configs and key not in self.default_configs.keys():
                print(os.linesep+"Warning: Config '{}' not supported!".format(key)+os.linesep)
        return configs_dict

    def __get_logger_recorder(self):
        logger=logging.getLogger("logger_{}".format(self.configs.name))
        logger.setLevel(logging.INFO)
        logger.handlers = []
        disk_handler = logging.FileHandler(filename=self.project_path+"training_event.log", mode='a')
        disk_handler.setFormatter(logging.Formatter(fmt="%(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(disk_handler)
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
        logger.addHandler(screen_handler)

        os.makedirs(self.records_path,exist_ok=True)
        recorder=SummaryWriter(log_dir=self.records_path)
        
        return logger,recorder

    def __get_optimizer(self,network):
        if self.configs.optimizer=="AdamW":
            return torch.optim.AdamW(network.parameters(),lr=self.configs.lr)
        elif self.configs.optimizer=="Adam":
            return torch.optim.Adam(network.parameters(),lr=self.configs.lr)
        elif self.configs.optimizer=="SGD":
            return torch.optim.SGD(network.parameters(),lr=self.configs.lr)
        else:
            raise ValueError("Optimizer '{}' not supported".format(self.configs.optimizer))

    def __get_lr_scheduler(self,optimizer):
        if self.configs.lr_scheduler=="cosine":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_cosine_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="linear":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_linear_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_constant_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        else:
            raise ValueError("Learning rate scheduler '{}' not supported".format(self.configs.lr_scheduler))

    def __train(self,network:torch.nn.Module,train_dataset,validation_dataset=None):
        #set random seed  
        torch.manual_seed(self.configs.random_seed)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        dataloader_genrator = torch.Generator()
        dataloader_genrator.manual_seed(self.configs.random_seed)
        # create project folder
        time_label = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if not self.__train_from_checkpoint:
            self.project_path=self.configs.save_path+self.configs.name+os.sep+time_label+os.sep
            os.makedirs(self.project_path, exist_ok=True)
        self.records_path=self.project_path+"records"+os.sep
        self.checkpoints_path=self.project_path+"checkpoints"+os.sep
        os.makedirs(self.checkpoints_path, exist_ok=True)
        #get logger and recorder
        self.logger,self.recorder=self.__get_logger_recorder()
        self.logger.info("Trainer created at {}".format(time_label))
        if self.__train_from_checkpoint:
            self.logger.info("Training from checkpoint, checkpoint epoch:{}".format(self.start_epoch))
        self.logger.info("Working path:{}".format(self.project_path))
        self.logger.info("Random seed: {}".format(self.configs.random_seed))      
        # save configs if not train from checkpoint
        if not self.__train_from_checkpoint:
            with open(self.project_path+"configs.yaml","w") as f:
                yaml.dump(self.__prepare_config_dict_for_inoutput(self.configs.__dict__,output=True),f)
        self.logger.info("Training configurations saved to {}".format(self.project_path+"configs.yaml"))
        # show model paras and save model structure
        self.logger.info("Network has {} trainable parameters".format(show_paras(network,print_result=False)))
        torch.save(network, self.project_path + "network_structure.pt")
        # get dataloader
        self.train_dataloader = DataLoader(train_dataset, 
                                           batch_size=self.configs.batch_size_train, 
                                           shuffle=self.configs.shuffle_train,
                                           num_workers=self.configs.num_workers_train,
                                           worker_init_fn=seed_worker,
                                           generator=dataloader_genrator
                                           )
        num_batches_train=len(self.train_dataloader)
        self.logger.info("There are {} training batches in each epoch".format(num_batches_train))
        self.logger.info("Batch size for training:{}".format(self.configs.batch_size_train))
        self.logger.info("Training epochs:{}".format(self.configs.epochs-self.start_epoch+1))
        self.logger.info("Total training iterations:{}".format(len(self.train_dataloader)*(self.configs.epochs-self.start_epoch+1)))
        if validation_dataset is not None:
            self.validate_dataloader = DataLoader(validation_dataset, 
                                                  batch_size=self.configs.batch_size_validation, 
                                                  shuffle=self.configs.shuffle_validation,
                                                  num_workers=self.configs.num_workers_validation,
                                                  worker_init_fn=seed_worker,
                                                  generator=dataloader_genrator
                                                  )
            num_batches_validation=len(self.validate_dataloader)
            self.logger.info("Validation will be done every {} epochs".format(self.configs.validation_epoch_frequency))
            self.logger.info("Batch size for validation:{}".format(self.configs.batch_size_validation))
        # set optimizer and lr scheduler
        self.optimizer = self.__get_optimizer(network)
        self.lr_scheduler = self.__get_lr_scheduler(self.optimizer)
        self.logger.info("learning rate:{}".format(self.configs.lr))
        self.logger.info("Optimizer:{}".format(self.configs.optimizer))
        self.logger.info("Learning rate scheduler:{}".format(self.configs.lr_scheduler))
        if self.configs.warmup_epoch!=0:
            self.logger.info("Use learning rate warm up, warmup epoch:{}".format(self.configs.warmup_epoch))
        if self.__train_from_checkpoint:
            checkpoint_file_path=self.checkpoints_path+"checkpoint_{}.pt".format(self.start_epoch-1)
            self.logger.info("Loading checkpoint from {}".format(checkpoint_file_path))
            checkpoint=torch.load(checkpoint_file_path)
            network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # main train loop
        if self.configs.record_iteration_loss:
            loss_tag="Loss_epoch"
        else:
            loss_tag="Loss"
        network.to(self.configs.device)
        self.logger.info("Training start!")
        p_bar=tqdm(range(self.start_epoch,self.configs.epochs+1))
        self.event_before_training(network)
        for idx_epoch in p_bar:
            train_losses_epoch=[]
            lr_now=self.optimizer.param_groups[0]["lr"]
            info_epoch="lr:{:.3e}".format(lr_now)
            self.recorder.add_scalar("learning_rate",lr_now,idx_epoch)
            network.train()
            for idx_batch,batched_data in enumerate(self.train_dataloader):
                loss = self.train_step(network=network, batched_data=batched_data,idx_batch=idx_batch,num_batches=num_batches_train,idx_epoch=idx_epoch,num_epoch=self.configs.epochs)
                self.back_propagate(loss,self.optimizer)
                train_losses_epoch.append(loss.item())
                if self.configs.record_iteration_loss:
                    self.recorder.add_scalar("Loss_iteration/train",train_losses_epoch[-1],(idx_epoch-1)*num_batches_train+idx_batch)
                self.event_after_training_iteration(network,idx_epoch,idx_batch)
            train_losses_epoch_average=sum(train_losses_epoch)/len(train_losses_epoch)
            info_epoch+=" train loss:{:.5f}".format(train_losses_epoch_average)
            self.recorder.add_scalar("{}/train".format(loss_tag),train_losses_epoch_average,idx_epoch)
            self.event_after_training_epoch(network,idx_epoch)
            if validation_dataset is not None and idx_epoch%self.configs.validation_epoch_frequency==0:
                validation_losses_epoch=[]
                network.eval()
                with torch.no_grad():
                    for idx_batch,batched_data in enumerate(self.validate_dataloader):
                        loss_validation=self.validation_step(network=network,batched_data=batched_data,idx_batch=idx_batch,num_batches=num_batches_validation,idx_epoch=idx_epoch,num_epoch=self.configs.epochs)
                        validation_losses_epoch.append(loss_validation.item())
                        if self.configs.record_iteration_loss:
                            self.recorder.add_scalar("Loss_iteration/validation",validation_losses_epoch[-1],(idx_epoch-1)*num_batches_validation+idx_batch)
                        self.event_after_validation_iteration(network,idx_epoch,idx_batch)
                    validation_losses_epoch_average=sum(validation_losses_epoch)/len(validation_losses_epoch)
                    info_epoch+=" validation loss:{:.5f}".format(validation_losses_epoch_average)
                    self.recorder.add_scalar("{}/validation".format(loss_tag),validation_losses_epoch_average,idx_epoch)
                    self.event_after_validation_epoch(network,idx_epoch)
            p_bar.set_description(info_epoch)
            self.lr_scheduler.step()
            if idx_epoch%self.configs.save_epoch==0:
                checkpoint_now={
                    "epoch":idx_epoch,
                    "network":network.state_dict(),
                    "optimizer":self.optimizer.state_dict(),
                    "lr_scheduler":self.lr_scheduler.state_dict()
                }
                torch.save(checkpoint_now,self.checkpoints_path+"checkpoint_{}.pt".format(idx_epoch))
        self.event_after_training(network)
        network.to("cpu")
        torch.save(network.state_dict(),self.project_path+"trained_network_weights.pt")
        self.logger.info("Training finished!")

    def set_configs_type(self):
        # mandatory configs
        self.mandatory_configs=[
            "name",
            "save_path",
            "batch_size_train",
            "epochs",
            "lr",
            ]
        # optional configs with default values
        self.default_configs={
            "device":"cpu",
            "random_seed":None,
            "batch_size_validation":None,
            "shuffle_train":True,
            "shuffle_validation":None,
            "num_workers_train":0,
            "num_workers_validation":None,
            "validation_epoch_frequency":1,
            "optimizer":"AdamW",
            "lr_scheduler":"cosine",
            "final_lr":None,
            "warmup_epoch":0,
            "save_epoch":None,
            "record_iteration_loss":False
                              }
        # configs that only allowed to be set to specific values
        self.configs_option={
            "optimizer":["AdamW","Adam","SGD"],
            "lr_scheduler":["cosine","linear","constant"],
        }

    def train_step(self,network:torch.nn.Module,batched_data,idx_batch:int,num_batches:int,idx_epoch:int,num_epoch:int):
        '''
        Args:
            network: torch.nn.Module, the network to be trained
            batched_data: torch.Tensor or tuple of torch.Tensor, the data for training, need to be moved to the device first!
            idx_batch: int, index of the current batch
            num_batches: int, total number of batches
            idx_epoch: int, index of the current epoch
            num_epoch: int, total number of epochs
        '''
        inputs=batched_data[0].to(self.configs.device)
        targets=batched_data[1].to(self.configs.device)
        predictions=network(inputs)
        loss=torch.nn.functional.mse_loss(predictions,targets)
        return loss
    
    def validation_step(self,network:torch.nn.Module,batched_data,idx_batch:int,num_batches:int,idx_epoch:int,num_epoch:int):
        '''
        Args:
            network: torch.nn.Module, the network to be trained
            batched_data: torch.Tensor or tuple of torch.Tensor, the data for validation,need to be moved to the device first!
            idx_batch: int, index of the current batch
            num_batches: int, total number of batches
            idx_epoch: int, index of the current epoch
            num_epoch: int, total number of epochs
        '''
        inputs=batched_data[0].to(self.configs.device)
        targets=batched_data[1].to(self.configs.device)
        predictions=network(inputs)
        loss=torch.nn.functional.mse_loss(predictions,targets)
        return loss

    def train_from_scratch(self,network,train_dataset,validation_dataset,path_config_file:str="",**kwargs):
        '''
        network: torch.nn.Module, the network to be trained, mandatory
        train_dataset: torch.utils.data.Dataset, the training dataset, mandatory
        validation_dataset: torch.utils.data.Dataset, the validation dataset, default is None. If None, no validation will be done.
        path_config_file: str, path to the yaml file of the training configurations, default is ""
        kwargs: dict, the training configurations, default is {}, will overwrite the configurations in the yaml file.
        
        Mandatory configurations for training:
            name: str, name of the training, mandatory
            save_path: str, path to save the training results, mandatory
            batch_size_train: int, batch size for training, mandatory
            epochs: int, number of epochs for training, mandatory
            lr: float, initial learning rate, mandatory
        Optional Configurations for training:
            device: cpu or cuda, device for training, default is cpu
            random_seed: None or int, random seed for training, default is None   
            batch_size_validation: int, batch size for validation, default is the same as batch_size_train
            shuffle_train: bool, whether to shuffle the training dataset, default is True
            shuffle_validation: bool, whether to shuffle the validation dataset, default is the same as shuffle_train
            num_workers_train: int, number of workers for training, default is 0
            num_workers_validation: int, number of workers for validation, default is the same as num_workers_train
            validation_epoch_frequency: int, frequency of validation, default is 1  
            optimizer: str, optimizer for training, default is AdamW
            lr_scheduler: str, learning rate scheduler for training, default is cosine   
            final_lr: float, final learning rate, default is the same as lr
            warmup_epoch: int, number of epochs for learning rate warm up, default is 0
            record_iteration_loss: bool, whether to record iteration loss, default is False
            save_epoch: int, frequency of saving checkpoints, default is 1/10 of epochs
        '''
        self.__train_from_checkpoint=False
        configs_dict=copy.deepcopy(self.default_configs)
        # read configs from yaml file
        if path_config_file != "":
            with open(path_config_file,"r") as f:
                yaml_configs=yaml.safe_load(f)
            for key in yaml_configs.keys():
                configs_dict[key]=yaml_configs[key]
        # read configs from kwargs
        for key in kwargs.keys():
            configs_dict[key]=kwargs[key]
        configs_dict=self.__check_configs_dict(configs_dict)
        self.configs=GeneralDataClass(self.__prepare_config_dict_for_inoutput(configs_dict,output=False))
        self.__train(network,train_dataset,validation_dataset)

    def train_from_checkpoint(self,project_path,train_dataset,validation_dataset,restart_epoch=None):
        '''
        project_path: str, path to the project folder, mandatory
        train_dataset: torch.utils.data.Dataset, the training dataset, mandatory
        validation_dataset: torch.utils.data.Dataset, the validation dataset, default is None. If None, no validation will be done.
        restart_epoch: int, the epoch to restart training, default is None, which means the latest checkpoint will be used.
        '''
        self.__train_from_checkpoint=True
        self.project_path=project_path
        # get checkpoint epoch
        if self.project_path[-1]!=os.sep:
            self.project_path+=os.sep
        if not os.path.exists(self.project_path+"configs.yaml"):
            print("No configs.yaml found in {}".format(self.project_path))
            print("Trying to use the latest subfolder as project path")
            dir_list = [folder_name for folder_name in os.listdir(self.project_path) if os.path.isdir(self.project_path+folder_name)]
            folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(self.project_path, x)))[-1]
            if not os.path.exists(self.project_path+folder_name+os.sep+"configs.yaml"):
                raise FileNotFoundError("No configs.yaml found in {}".format(self.project_path+folder_name+os.sep))
            self.project_path+=folder_name+os.sep
            print("Project path set to {}".format(self.project_path))
        if restart_epoch is None:
            check_points_names=os.listdir(self.project_path+"checkpoints")
            latest_check_point_name = sorted(check_points_names,  key=lambda x: os.path.getmtime(os.path.join(self.project_path+"checkpoints"+os.sep, x)))[-1]
            restart_epoch=int(latest_check_point_name.split("_")[-1].split(".")[0])
        # check files
        if not os.path.exists(self.project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(restart_epoch)):
            raise FileNotFoundError("No 'checkpoint_{}.pt' found in {}".format(restart_epoch,self.project_path+"checkpoints"+os.sep))
        if not os.path.exists(self.project_path+"network_structure.pt"):
            raise FileNotFoundError("No network_structure.pt found in {}".format(self.project_path))
        # read configs and network
        with open(self.project_path+"configs.yaml","r") as f:
            configs_dict=yaml.safe_load(f)
            configs_dict=self.__check_configs_dict(configs_dict)
        self.configs=GeneralDataClass(self.__prepare_config_dict_for_inoutput(configs_dict,output=False))
        network=torch.load(self.project_path+"network_structure.pt")
        self.start_epoch=restart_epoch+1
        self.__train(network,train_dataset,validation_dataset)

    def back_propagate(self,loss:torch.Tensor,optimizer:torch.optim.Optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def event_before_training(self,network):
        pass
    
    def event_after_training(self,network):
        pass
    
    def event_after_training_epoch(self,network,idx_epoch):
        pass
    
    def event_after_training_iteration(self,network,idx_epoch,idx_batch):
        pass
    
    def event_after_validation_epoch(self,network,idx_epoch):
        pass
    
    def event_after_validation_iteration(self,network,idx_epoch,idx_batch):
        pass

class TrainedProject():
    
    def __init__(self,project_path) -> None:
        if project_path[-1]!=os.sep:
            project_path+=os.sep
        if not os.path.exists(project_path+"network_structure.pt"):
            print("Warning: No network structure found in {}".format(project_path),flush=True)
            print("Trying to use the latest subfolder as project path",flush=True)
            dir_list = [folder_name for folder_name in os.listdir(project_path) if os.path.isdir(project_path+folder_name)]
            folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(project_path, x)))[-1]
            if not os.path.exists(project_path+folder_name+os.sep+"network_structure.pt"):
                raise FileNotFoundError("No configs.yaml found in {}".format(project_path+folder_name+os.sep))
            project_path+=folder_name+os.sep
        self.project_path=project_path
    
    def get_configs(self,only_path=False):
        if not os.path.exists(self.project_path+"configs.yaml"):
            raise FileNotFoundError("No configs.yaml found in {}".format(self.project_path))
        if only_path:
            return self.project_path+"configs.yaml"
        else:
            with open(self.project_path+"configs.yaml","r") as f:
                configs_dict=yaml.safe_load(f)
            return configs_dict
    
    def get_network_strcuture(self,only_path=False):
        if not os.path.exists(self.project_path+"network_structure.pt"):
            raise FileNotFoundError("No network_structure.pt found in {}".format(self.project_path))
        if only_path:
            return self.project_path+"network_structure.pt"
        else:
            return torch.load(self.project_path+"network_structure.pt")
    
    def get_trained_network_weights(self,only_path=False):
        if not os.path.exists(self.project_path+"trained_network_weights.pt"):
            raise FileNotFoundError("No trained_network_weights.pt found in {}".format(self.project_path))
        if only_path:
            return self.project_path+"trained_network_weights.pt"
        else:
            return torch.load(self.project_path+"trained_network_weights.pt")
    
    def get_checkpoints(self,only_path=False,check_point=None):
        if check_point is not None:
            if not os.path.exists(self.project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(check_point)):
                check_points_names=[case.split("_")[1].split(".")[0] for case in os.listdir(self.project_path+"checkpoints")]
                raise FileNotFoundError("No 'checkpoint_{}.pt' found in {}".format(check_point,self.project_path+"checkpoints"+os.sep)
                                        +os.linesep+"Possible checkpoints:"+os.linesep+str(check_points_names))
            check_point_path=self.project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(check_point)
        else:
            print("No checkpoint specified, using the latest checkpoint",flush=True)
            print("Trying to load checkpoints from the latest checkpoint",flush=True)
            check_points_names=os.listdir(self.project_path+"checkpoints")
            latest_check_point_name = sorted(check_points_names,  key=lambda x: os.path.getmtime(os.path.join(self.project_path+"checkpoints"+os.sep, x)))[-1]
            check_point_path=self.project_path+"checkpoints"+os.sep+latest_check_point_name
        if only_path:
            return check_point_path
        else:
            return torch.load(check_point_path)
    
    def get_records(self,only_path=False):
        records=os.listdir(self.project_path+"records"+os.sep)
        if len(records)==0:
            raise FileNotFoundError("No records found in {}".format(self.project_path+"records"+os.sep))
        records_path=self.project_path+"records"+os.sep+records[0]
        if only_path:
            return records_path
        else:
            ea= event_accumulator.EventAccumulator(records_path)
            ea.Reload()
            return ea
    
    def get_saved_net(self,check_point=None):
        network=self.get_network_strcuture(only_path=False)
        if check_point is not None:
            weights=self.get_checkpoints(check_point=check_point)["network"]
        else:
            if os.path.exists(self.project_path+"trained_network_weights.pt"):
                weights=torch.load(self.project_path+"trained_network_weights.pt")        
            else:
                print("Warning: No trained_network_weights.pt found in {}".format(self.project_path),flush=True)
                weights=self.get_checkpoints(check_point=None)["network"]
        network.load_state_dict(weights)
        return network

def read_configs(path_config_file:str=""):
    '''
    Read the training configurations from a yaml file.
    
    Args:
        path_config_file: str, path to the yaml file of the training configurations, default is ""
    '''
    if path_config_file != "":
        with open(path_config_file,"r") as f:
            yaml_configs=yaml.safe_load(f)
        return yaml_configs
    else:
        return None

def get_saved_net(project_path,check_point=None):
    return TrainedProject(project_path).get_saved_net(check_point=check_point)
    '''
    if project_path[-1]!=os.sep:
        project_path+=os.sep
    if not os.path.exists(project_path+"network_structure.pt"):
        print("Warning: No network structure found in {}".format(project_path),flush=True)
        print("Trying to use the latest subfolder as project path",flush=True)
        dir_list = [folder_name for folder_name in os.listdir(project_path) if os.path.isdir(project_path+folder_name)]
        folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(project_path, x)))[-1]
        if not os.path.exists(project_path+folder_name+os.sep+"network_structure.pt"):
            raise FileNotFoundError("No configs.yaml found in {}".format(project_path+folder_name+os.sep))
        project_path+=folder_name+os.sep
    if check_point is not None:
        network_weights=torch.load(project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(check_point))["network"]
    else:
        if os.path.exists(project_path+"trained_network_weights.pt"):
            network_weights=torch.load(project_path+"trained_network_weights.pt")
        else:
            print("Warning: No trained_network_weights.pt found in {}".format(project_path),flush=True)
            print("Trying to load weights from the latest checkpoint",flush=True)
            check_points_names=os.listdir(project_path+"checkpoints")
            latest_check_point_name = sorted(check_points_names,  key=lambda x: os.path.getmtime(os.path.join(project_path+"checkpoints"+os.sep, x)))[-1]
            network_weights=torch.load(project_path+"checkpoints"+os.sep+latest_check_point_name)["network"]        
    network=torch.load(project_path+"network_structure.pt")
    network.load_state_dict(network_weights)
    return network
    '''


