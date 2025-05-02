"""
optilab
========

A python library to test optimization
"""

__version__ = "0.1.0"

# Set the default device to cuda if possible
import torch 
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(dev)


# Set double precision
torch.set_default_dtype(torch.float64)

from src.models import *
from src.optimizers import *
from src.utils import *
from src.testing import * 
from src.initializers import * 
from src.PDEs_ScikitFEM import *
from src.models_configurated import *
