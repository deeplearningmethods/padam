# PADAM
This repository includes the code used for the numerical experiments of the preprint '[*PADAM: Parallel averaged Adam reduces the error for stochastic optimization in scientific machine learning*](https://arxiv.org/abs/2505.22085)' by Arnulf Jentzen, Julian Kranz, and Adrian Riekert.

Our experiments were conducted on an Ubuntu 24.04.1 LTS system with an Nvidia RTX3090 GPU using Python 3.12.
To install the required Python dependencies, run 
```
pip install torch numpy tqdm pandas scikit-fem matplotlib
```
in the terminal. 

The conducted experiments can be reproduced by executing the python files from the main directory. 
For example, to reproduce the experiments with the PINN for an Allen--Cahn equation, run 
```
python PADAM_allencahn.py
```
from the main directory. The resulting data will be stored as a `.pkl`-file in `tests/data/` and in `tests/datacsv/` as `.csv`-files. The created file names are unique for each execution.
Upon execution, the plots appearing in the paper are automatically created as `.pdf`-files in `tests/plots/`. 
The `.pkl`-files can be loaded into python using the filename and the `load_result_from_file` as follows:
```
python 
>>> from src import *
>>> result = load_result_from_file("Allen Cahn PINN_[3, 32, 64, 32, 1]_174300226350547")
```
The object `result` is then an instance of the `OptimizationResults` class and can be manipulated with various methods implemented in the file `src/testing.py`.

***

This work has been supported by the Ministry of Culture and Science NRW as part of the Lamarr Fellow Network.
In addition, we gratefully acknowledge the Cluster of Excellence Mathematics Münster

License: https://creativecommons.org/licenses/by/4.0/