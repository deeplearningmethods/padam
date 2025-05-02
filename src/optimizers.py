""" This file contains the optimizers for the experiments """

# These two lines allow us to import modules for typechecking without actually using them 
from __future__ import annotations
from typing import TYPE_CHECKING 
from typing import Callable

from dataclasses import dataclass, field 
import torch
if TYPE_CHECKING:
    from src.models import Model 
from src.testing import OptimizationResult
from tqdm import tqdm 
import math 
from src.utils import *
import copy
from time import time


def intitialize(optimize: Callable):
    """"
        A decorator that can be placed before the optimizer.optimize method in order to initialize the model before optimization
    """
    # We have to pass optim as an argument as well since it acts as the self argument
    def wrapper(optim, model,*args,**kwargs) -> OptimizationResult:
        model.initialize_weights()
        out = optimize(optim, model,*args,**kwargs)
        return out 
    return wrapper 

@dataclass
class Optimizer:
    """
        abstract optimizer class from which everything else inherits
    """
    nr_steps: int = 20000 # How many steps you want to optimize
    bs: int = 32 # Batch size for optimization
    lr: float = 0.001 # Learning rate
    n_test: int = 1 # Batch size for evaluating
    eval_points: int = 200 # how many times you want to evaluate (spread evenly through the steps)
    with_norm: bool = True # Do you wanna plot the norms of the model as well?
    id: str = "opt" # unique identifier for the optimizer (not used at the moment)
    plotcolor: str = "C0" # Color in which you want to plot the result 
    plotlabel: str = "opt" # label which you want to see in the plot 
    linestyle: str = "solid" # linestyle for the plot 
    linewidth: float = 1.0 # linewidth
    runs: int = 1 # number of independent runs. If >1, you will plot the mean and optionally the variance of the independent runs. 

    def __post_init__(self) -> None:
        self.eval_steps = self.nr_steps // self.eval_points

    def optimize(self, model: Model) -> OptimizationResult:
        """ This method runs nr_steps many optimization steps with the given settings on the given model and stores it in an OptimizationResult."""
        pass 

    def optimize_several(self, model: Model, runs: int = None) -> OptimizationResult:
        """ This method runs self.optimize self.runs many times and stores the result """
        out = OptimizationResult(model, self, data=[]) # create an empty optimization result
        if runs is not None:
            self.runs = runs 
        for i in range(self.runs):
            print(f"Run {i+1} / {self.runs}")
            model.initialize_weights() # not necessary since the optimizers are decorated with @initialize
            result = self.optimize(model)
            out.append(result)
        return out 
    
    def calc_norm_diffs(self, norm_list: list[torch.Tensor]) -> list[float]:
        """
            Given a list of parameter vectors of the ANN, this outputs a list of norm differences of the parameter vectors to the last parameter vector.
            If with_norm == False, this just outputs a list of zeros
            when quadratic is set to True, this tracks the quadratic model 
        """
        with torch.no_grad():
            norm_diffs = []
            for l in norm_list:
                sum2 = 0.
                if self.with_norm:
                    for p, q in zip(l, norm_list[-1]):
                        if True: # If this is set to True, we calculate the norms.
                            q = 0*q
                        sum2 += (p - q).square().sum().cpu().numpy().item()
                norm_diffs.append(math.sqrt(sum2))
            return norm_diffs 


@dataclass 
class plain_optimizer(Optimizer):
    """
        This class takes a torch.optim instance (standard choice: torch.optim.SGD) and runs optimization on this 
    """
    id: str = "sgd"
    plotlabel: str = "SGD"
    plotcolor: str = "C0"
    optim: torch.optim.Optimizer = torch.optim.SGD
                 
    
    
         
    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        optimizer = self.optim(model.parameters(),lr=self.lr)
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            optimizer.zero_grad()
            data = model.initializer.sample(self.bs)
            loss = model.loss(data)
            loss.backward()
            optimizer.step()
            steps += 1

            if (n + 1) % self.eval_steps == 0: 
                step_list.append(steps)
                test_data = model.initializer.sample(self.n_test)
                train_losses.append(model.loss(test_data).cpu().detach().numpy().item())
                errors.append(model.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(optimizer.lr)
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        return result 

@dataclass 
class SGD(plain_optimizer):
    plotlabel: str = "SGD"
    plotcolor: str = "C0"
    optim: torch.optim = torch.optim.SGD

@dataclass
class SGDM(Optimizer):
    """
        SGD with momentum  
    """
    id: str = "sgdm"
    plotlabel: str = "SGDm"
    plotcolor: str = "C6"
    optim: torch.optim.Optimizer = torch.optim.SGD
                 
    
    
         
    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        optimizer = self.optim(model.parameters(),lr=self.lr, momentum = 0.9)
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            optimizer.zero_grad()
            data = model.initializer.sample(self.bs)
            loss = model.loss(data)
            loss.backward()
            optimizer.step()
            steps += 1

            if (n + 1) % self.eval_steps == 0: 
                step_list.append(steps)
                test_data = model.initializer.sample(self.n_test)
                train_losses.append(model.loss(test_data).cpu().detach().numpy().item())
                errors.append(model.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(optimizer.lr)
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        return result 

def _betas():
    return [0.9,0.999]

@dataclass
class ADAM(plain_optimizer):
    plotlabel: str = "ADAM"
    plotcolor: str = "C1"
    optim: torch.optim = torch.optim.Adam
    betas: list[float] = field(default_factory=_betas)

    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        optimizer = torch.optim.Adam(model.parameters(),lr=self.lr,betas= self.betas )
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            optimizer.zero_grad()
            data = model.initializer.sample(self.bs)
            loss = model.loss(data)
            loss.backward()
            optimizer.step()
            steps += 1

            if (n + 1) % self.eval_steps == 0: 
                step_list.append(steps)
                test_data = model.initializer.sample(self.n_test)
                train_losses.append(model.loss(test_data).cpu().detach().numpy().item())
                errors.append(model.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(optimizer.lr)
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        return result 
    
@dataclass 
class ADAMW(plain_optimizer):
    plotlabel: str = "ADAMW"
    plotcolor: str = "C3"
    optim: torch.optim = torch.optim.AdamW

@dataclass 
class arithm_average_adam_from_start(Optimizer):
    """
        first runs an ADAM optimizer and from the point `start` onwards, it just averages all the previous parameters 
    """
    id: str = "arithmetic-average-adam-fromstart"
    plotlabel: str = "arithmetic-average-ADAM-fromstart"
    plotcolor: str = "C0"
    start: int = 1
    

            
    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        """
            This creates a copy `copied_model` of the model, then optimizes this model using the Adam optimizer. 
            The actual model is first copied, and after the point `start`, it just averages all the parameters 
            of the copied model from that point on  
        """

        copied_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(copied_model.parameters(),lr=self.lr)
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            # Perform an Adam step with the copied model 
            optimizer.zero_grad()
            data = copied_model.initializer.sample(self.bs)
            loss = copied_model.loss(data)
            loss.backward()
            optimizer.step()
            # copy the average of the last 1000 iterates into the model itself 
            with torch.no_grad():
                if n < self.start:
                    for p, pp in zip(copied_model.parameters(), model.parameters()):
                        pp.copy_(p)
                else:
                    for p, pp in zip(copied_model.parameters(), model.parameters()):
                        pp.mul_((n+1 - self.start))
                        pp.add_(p)
                        pp.mul_(1. / (n + 2 - self.start))
                steps += 1

            if (n + 1) % self.eval_steps == 0: 
                step_list.append(steps)
                test_data = model.initializer.sample(self.n_test)
                train_losses.append(model.loss(test_data).cpu().detach().numpy().item())
                errors.append(model.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(optimizer.lr)
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        return result 
    
@dataclass 
class geom_average_adam(Optimizer):
    """
        geometric average adam 
    """
    id: str = "geometric-average-adam"
    plotlabel: str = "geometric-average-ADAM"
    plotcolor: str = "C0"
    gamma: float = 0.999

            
    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        """
           
        """
        copied_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(copied_model.parameters(),lr=self.lr)
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            # Perform an Adam step with the copied model 
            optimizer.zero_grad()
            data = copied_model.initializer.sample(self.bs)
            loss = copied_model.loss(data)
            loss.backward()
            optimizer.step()
            # averaging
            with torch.no_grad():
                for p, pp in zip(copied_model.parameters(), model.parameters()):
                    pp.mul_(self.gamma)
                    pp.add_((1. - self.gamma) * p)
                steps += 1

            if (n + 1) % self.eval_steps == 0: 
                step_list.append(steps)
                test_data = model.initializer.sample(self.n_test)
                train_losses.append(model.loss(test_data).cpu().detach().numpy().item())
                errors.append(model.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(optimizer.lr)
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        return result 
    
@dataclass 
class PADAM3(Optimizer):
    """
        PADAM with the 3 best channels, including vanilla Adam.
        when test_steps is None, this implementation logs the best model at every single testing step. 
        otherwise, it just checks every test_steps many steps. 
    """
    id: str = "PADAM3"
    plotlabel: str = "PADAM3"
    plotcolor: str = "C0"
    test_steps: int = None #5000
    channels:int=4
    logtesttime:bool=True # logging how much time it takes to test the channels 



    
    def find_best_model(self, model_list:list)-> Tuple[Model, int]:
        """
            finds the best model in a list of models. 
        """
        start = time()
        index = 0
        best_model = model_list[0]
        best_loss = torch.inf
        for i, mod in enumerate(model_list):
            data = mod.initializer.sample(self.n_test)
            loss = mod.test_loss(data)
            if loss < best_loss:
                best_model = mod 
                best_loss = loss
                index = i 
        end = time()
        if self.logtesttime:
            self.testtime += end-start 
        return best_model, index
    def schedule_list(self,i):
        schedule_list = [
                lambda n:0., #This is the model we optimize with Adam
                lambda n:0.999,
                lambda n: 1.-(n+1)**(-0.7),   
                lambda n:  1.-math.exp(math.log((1.-0.999)/(1.-0.9))/self.nr_steps * n)* (1. - 0.9)                          
                                                                   ]
        return schedule_list[i]
    
    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        """
           optimize
        """
        # if no test_steps are given, we switch to the best model at every single evaluation
        if self.test_steps is None:
            self.test_steps = self.eval_steps

        if self.logtesttime:
            self.testtime = 0.
        averaged_models = [copy.deepcopy(model) for _ in range(self.channels)] # The averaged models 
        # this is the model where we perform Adam: 
        ref_model = copy.deepcopy(model)#averaged_models[0]
        index_log=0
        # In the beginning, we declare the reference model to be the best model.
        model = ref_model 
        # define the base optimizer
        optimizer = torch.optim.Adam(ref_model.parameters(),lr=self.lr)
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            # Perform an Adam step with the base model 
            optimizer.zero_grad()
            data = ref_model.initializer.sample(self.bs)
            loss = ref_model.loss(data)
            loss.backward()
            optimizer.step()
            steps += 1
            
            # Find the best model 
            if (n+1) % self.test_steps == 0:
                model_log, index_log = self.find_best_model(averaged_models)
                model = model_log # save the best parameters in the actual model you optimize
                # Here we can choose the best model. 
            
            # averaging the different averaged models 
            with torch.no_grad():
                for i,av_mod in enumerate(averaged_models):
                    for p, pp in zip(ref_model.parameters(), av_mod.parameters()):
                        pp.mul_(self.schedule_list(i)(n))
                        pp.add_((1. - self.schedule_list(i)(n)) * p)
            
                
            if (n + 1) % self.eval_steps == 0: 
                # calculate best model and gamma for plotting 
                step_list.append(steps)
                test_data = model.initializer.sample(self.n_test)
                train_losses.append(model.loss(test_data).cpu().detach().numpy().item())
                errors.append(model.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(index_log) # we log the channel in the learning rate part.
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        result.testtime = self.testtime
        return result 
     
@dataclass 
class PADAM10(Optimizer):
    """
        PADAM with the 10 best channels, including vanilla Adam.
        when test_steps is None, this implementation logs the best model at every single testing step. 
        otherwise, it just checks every test_steps many steps. 
    """
    id: str = "PADAM10"
    plotlabel: str = "PADAM10"
    plotcolor: str = "C0"
    test_steps: int = None #5000
    channels:int=11
    logtesttime:bool = True 


    
    def find_best_model(self, model_list:list)-> Tuple[Model, int]:
        """
            finds the best model in a list of models. 
        """
        start = time()
        index = 0
        best_model = model_list[0]
        best_loss = torch.inf
        for i, mod in enumerate(model_list):
            data = mod.initializer.sample(self.n_test)
            loss = mod.test_loss(data)
            if loss < best_loss:
                best_model = mod 
                best_loss = loss
                index = i 
        end = time()
        if self.logtesttime:
            self.testtime += end-start 
        return best_model, index
    def schedule_list(self,i):
        schedule_list = [
                lambda n:0., #This is the model we optimize with Adam
                lambda n:0.99,
                lambda n:0.999,
                lambda n: 1.-(n+1)**(-0.6), 
                lambda n: 1.-(n+1)**(-0.7),   
                lambda n: 1.-(n+1)**(-0.8),  
                lambda n: 1.-0.5*(n+1)**(0.7),
                lambda n:  1.-math.exp(math.log((1.-0.999)/(1.-0.9))/self.nr_steps * n)* (1. - 0.9),
                lambda n:  1.-math.exp(math.log((1.-0.999)/(1.-0.99))/self.nr_steps * n)* (1. - 0.99),
                lambda n:  1.-math.exp(math.log((1.-0.9999)/(1.-0.9))/self.nr_steps * n)* (1. - 0.9),      
                lambda n:  1.-math.exp(math.log((1.-0.999999)/(1.-0.9))/self.nr_steps * n)* (1. - 0.9)                 
        ]       
        return schedule_list[i] 
    
    @intitialize
    def optimize(self, model: Model) -> OptimizationResult:
        """
           optimize
        """
        # if no test_steps are given, we switch to the best model at every single evaluation
        if self.test_steps is None:
            self.test_steps = self.eval_steps
        if self.logtesttime:
            self.testtime = 0.
        averaged_models = [copy.deepcopy(model) for _ in range(self.channels)] # The averaged models 
        # this is the model where we perform Adam: 
        ref_model = copy.deepcopy(model) #averaged_models[0]
        # In the beginning, we declare the reference model to be the best model.
        model_log = ref_model
        index_log = 0
        # define the base optimizer
        optimizer = torch.optim.Adam(ref_model.parameters(),lr=self.lr)
        optimizer.lr = self.lr # storing the lr in the optimizer for plotting purposes 
        errors = []
        accs = [] # accuracies are not used at the time 
        lrs_list = [] # learning rates at evaluation time
        train_losses = []
        steps = 0
        norm_list = []
        step_list = []

        for n in tqdm(range(self.nr_steps)):
            # Perform an Adam step with the base model 
            optimizer.zero_grad()
            data = ref_model.initializer.sample(self.bs)
            loss = ref_model.loss(data)
            loss.backward()
            optimizer.step()
            steps += 1
            
            # Find the best model 
            if (n+1) % self.test_steps == 0:
                model_log, index_log = self.find_best_model(averaged_models)
                # Here we can choose the best model. 
            
            # averaging the different averaged models 
            with torch.no_grad():
                for i,av_mod in enumerate(averaged_models):
                    for p, pp in zip(ref_model.parameters(), av_mod.parameters()):
                        pp.mul_(self.schedule_list(i)(n))#
                        pp.add_((1. - self.schedule_list(i)(n) ) * p)#self.schedule_list(i)(n)
            
                
            if (n + 1) % self.eval_steps == 0: 
                # calculate best model and gamma for plotting 
                step_list.append(steps)
                test_data = model_log.initializer.sample(self.n_test)
                train_losses.append(model_log.loss(test_data).cpu().detach().numpy().item())
                errors.append(model_log.test_loss(test_data).cpu().detach().numpy().item())
                lrs_list.append(index_log) # we log the channel in the learning rate part.
                if self.with_norm:
                    norm_list.append([p.clone().detach() for p in model_log.parameters()])
                else:
                    norm_list.append(0)
        norm_diffs = self.calc_norm_diffs(norm_list)
        
        output = output_dict(step_list, lrs_list, train_losses, errors, norm_diffs)
        result = OptimizationResult(model,self,data=[output])
        result.testtime = self.testtime
        return result 