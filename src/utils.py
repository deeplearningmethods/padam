"""
This file contains mutliple helpful functions that are used in the code.
"""

# These two lines allow us to import modules for typechecking without actually using them 
from __future__ import annotations
from typing import TYPE_CHECKING 
from typing import Tuple
from typing import Callable

import torch
if TYPE_CHECKING:
    from src.models import Model
    import matplotlib

import matplotlib.pyplot as plt 
import math 
from time import time 
import numpy as np
import os


TEST_RESTULT_TABLE_PATH = "tests/tables/"
PAPER_PATH = "paper/"

TEST_RESULT_DATA_PATH = "tests/data/"
TEST_RESTULT_PLOT_PATH = "tests/plots/"
TEST_RESULT_LOG_PATH = "tests/logs/"
TEST_RESULT_DATACSV_PATH = "tests/datacsv/"

# helper function for creating new directory
def create_dir(dir:str) ->None:
    """
        helper function that creates a new directory
    """
    try:
        os.makedirs(dir, exist_ok=True)  # will not throw an error if the directory exists
    except Exception as e:
        print(f"An error occurred: {e}")

def which_device(model: Model) -> torch.device:
    """
        Determines, on which device the Model is stored on 
    """
    param = [j.device for j in model.parameters()]
    print(f"{model.id} is on {param[0]}")
    return param[0]

def give_device() -> torch.device:
    """
        returns torch.device("cuda") if available and torch.device("cpu") else.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def output_dict(step_list: list[int], lrs_list: list[float], train_losses: list[float], errors: list[float] , norm_diffs: list[float]) -> dict:
    """
        Takes the output of an optimization run and stores them in a dictionary
    """
    return {"step_list": step_list, "lrs": lrs_list, "train_losses": train_losses, "errors": errors, "norm_diffs": norm_diffs}


    
from time import time
def append_hash(name: str) -> str:
    """
        takes a string and appends a number that depends on the current time 
    """
    precision = 5 # how many decimal places do you want to use from the time since 1970 in seconds
    timestr = f"{time():.{precision}f}" # convert the time to str
    timestr = timestr[:-(precision +1)] + timestr[-precision:] # delete the decimal point
    hash = name + "_" + timestr # concatenate with the name 
    return hash

def gen_fig_ax(keys: dict[str], figsize=None) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
        Generates a matplotlib figure and axis with the appropriate size given a dictionary with keys (like "lrs", "errors",...)
    """
    num_plots = len(keys)
    if figsize is None:
        figsize = (8,8*num_plots)
    fig, ax = plt.subplots(num_plots, figsize=figsize)
    return fig, ax 


def parameter_norm(model) -> float:
    """
        Computes the L2 norm of the parameters of a model 
    """
    par = [p.clone().detach() for p in model.parameters()]
    sum2 = 0.
    for p in par:
        sum2 += p.square().sum().cpu().numpy().item()
    norm = math.sqrt(sum2)
    return norm 

def torchfunc_to_numpyfunc(g:Callable[[torch.tensor],torch.tensor])->Callable[[np.array],np.array]:
    """ Takes a function that is defined on torch tensors and returns the same function defined on numpy arrays"""
    def f(x:np.array) -> np.array:
        x=torch.tensor(x)
        return g(x).cpu().detach().numpy()
    return f



