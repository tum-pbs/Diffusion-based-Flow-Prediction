from .trainer_base import *
from .diffuser import *
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

class TrainerStepLr(Trainer):
    
    def __init__(self) -> None:
        super().__init__()

    def set_configs_type(self):
        super().set_configs_type()
        lr_scheduler_configs=self.configs_handler.get_config_features("lr_scheduler")
        lr_scheduler_configs["option"].append("step")
        self.configs_handler.set_config_features("lr_scheduler",lr_scheduler_configs)

    def get_lr_scheduler(self,optimizer):
        if self.configs.lr_scheduler=="step":
            if self.configs.warmup_epoch !=0:
                raise ValueError("Step Learning rate scheduler doesn't support warm up.")
            final_lr_ratio=self.configs.final_lr/self.configs.lr
            lr_change_frequency=0.01
            lr_gamma = math.pow(final_lr_ratio, 1/(1/lr_change_frequency-1))
            lr_step_size = int(lr_change_frequency*self.configs.epochs)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma, last_epoch=-1)
        else:
            return super().get_lr_scheduler(optimizer)

    def event_before_training(self, network):
        if not self._train_from_checkpoint:
            network.save_current_configs(self.project_path+"configs_network.yaml")

class DiffusionTrainer(TrainerStepLr):
    
    def __init__(self) -> None:
        super().__init__()
    
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("diffusion_step",value_type=int,default_value=200,description="The number of diffusion steps.")
    
    def event_before_training(self,network):
        # you can also use other diffuser, such as linear diffuser and sigmoid diffuser.
        self.diffuser = Cos2ParamsDiffuser(steps=self.configs.diffusion_step,device=self.configs.device)
        
    def train_step(self, network: torch.nn.Module, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        condition = batched_data[0].to(device=self.configs.device)
        targets = batched_data[1].to(device=self.configs.device)
        t = torch.randint(1, self.diffuser.steps+1,
                          size=(targets.shape[0],), device=self.configs.device)
        noise = torch.randn_like(targets, device=self.configs.device)
        xt = self.diffuser.forward_diffusion(targets, t, noise)
        predicted_noise = network(xt, t, condition)
        loss=torch.nn.functional.mse_loss(predicted_noise, noise)
        return loss

class BNNTrainer(TrainerStepLr):
    
    def __init__(self) -> None:
        super().__init__()
        
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("KL_scaling",value_type=float,default_value=0.01,description="The scaling factor of BNN training.")
    
    def train_step(self, network: torch.nn.Module, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        inputs = batched_data[0].to(device=self.configs.device)
        targets = batched_data[1].to(device=self.configs.device)
        prediction = network(inputs, torch.ones(size=(targets.shape[0],), device=self.configs.device)*200, None)
        mseloss=torch.nn.functional.mse_loss(prediction, targets)
        klloss=get_kl_loss(network)*((2**(num_batches-(idx_batch+1)))/(2**num_batches-1))*self.configs.KL_scaling
        self.recorder.add_scalar("Seprate_train_loss/mse",mseloss.item(),(idx_epoch-1)*num_batches+idx_batch)
        self.recorder.add_scalar("Seprate_train_loss/kl_loss",klloss.item(),(idx_epoch-1)*num_batches+idx_batch)
        return mseloss+klloss

class HeteroscedasticTrainer(TrainerStepLr):
    
    def __init__(self) -> None:
        super().__init__()
    
    def train_step(self, network: torch.nn.Module, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        inputs = batched_data[0].to(device=self.configs.device)    
        targets = batched_data[1].to(device=self.configs.device)
        prediction = network(inputs, torch.ones(size=(targets.shape[0],), device=self.configs.device)*200, None)
        heteroscedastic_loss=self.heteroscedastic_loss(prediction[:,0:3,:,:],prediction[:,3:6,:,:],targets)
        with torch.no_grad():
            mse=torch.nn.functional.mse_loss(prediction[:,0:3,:,:],targets)
            self.recorder.add_scalar("mse_loss",mse.item(),(idx_epoch-1)*num_batches+idx_batch)
        return heteroscedastic_loss

    def heteroscedastic_loss(self,prediction_mean,log_sig2,target):
        return torch.mean(0.5*torch.exp(-1*log_sig2)*torch.pow(prediction_mean-target,2)+0.5*log_sig2)