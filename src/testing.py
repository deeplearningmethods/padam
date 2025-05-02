"""
This file contains the code for systematically testing optimizers on models and for storing and visualizing the results. 
The results are stored as pickle files and can be loaded with the load_result_from_file routine 
"""

# These two lines allow us to import modules for typechecking without actually using them 
from __future__ import annotations
from typing import TYPE_CHECKING 
from typing import Tuple

from pandas import DataFrame
import pandas as pd
if TYPE_CHECKING:
    from src.optimizers import Optimizer 
    from src.models import Model
    import matplotlib 
    
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from src.utils import *
from time import time 



TEST_RESULT_DATA_PATH = "tests/data/"
TEST_RESTULT_PLOT_PATH = "tests/plots/"
TEST_RESULT_LOG_PATH = "tests/logs/"
TEST_RESULT_DATACSV_PATH = "tests/datacsv/"


class OptimizationResult:
    """
        a class that contains the results of multiple runs of a single optimizer on a single model
    """
    def __init__(self,
                 model: Model, 
                 optimizer: Optimizer,
                 data: list[dict] = [{
                    "step_list": [],
                    "lrs": [],
                    "train_losses": [],
                    "errors": [],
                    "norm_diffs": []
                    }]
                 ) -> None:
    
        self.data = [DataFrame(run) for run in data]
        self.model = model.plotname + "_" + str(model.neurons) # Avoiding issues with pickle, so only storing a string with some basic info 
        self.optimizer = optimizer
        self.filename = self.generate_filename()
        return 

    def append(self, result: OptimizationResult) -> OptimizationResult:
        """
            This appends the results of another OptimizationResult to the current data. 
            Only use this method if both self and result have the same model and optimizer.
        """
        self.data = [*self.data, *result.data]
        if hasattr(result,"testtime"):
            self.testtime = result.testtime
        if hasattr(result,"time"):
            self.time = result.time
        return self 
        
    def mean_variance(self) -> Tuple[DataFrame, DataFrame]:
        """
            Returns two DataFrames containing the means and the variances of list of runs 
        """
        combined_df = pd.concat(self.data, axis=0)
        mean_df = combined_df.groupby(combined_df.index).mean()
        var_df = combined_df.groupby(combined_df.index).var()
        return mean_df, var_df
    
    def plot(self, 
             fig = None, 
             ax = None, 
             keys: list[str] = ["errors"], #,"lrs"],# , "norm_diffs"], # ["lrs", "train_losses", "errors", "norm_diffs"], # select the data that you want to plot 
             scale: str = "log", # scale in which you want to plot. Usually "log" or "linear"
             heading: str = None, # You can choose a heading. Else, the function will automatically generate one.
             return_: bool = False, # Choose if you want to return fig, ax 
             savefig: bool = False, # Choose if you want to save the figure in tests/plots
             plotmode: str = "mean", # do you want to plot the "mean" or "all" paths? 
             shade_for_variance: float = 0.0, # do you want to shade the area above and below the line in the plot for multiple runs? enter number between 0.0 (no shade) and 1.0.
             legend: bool = True,
             grid:bool=True,
             bbox=(0.5, -0.15), # where do we put the legend
             loc='upper center', # where do we out the legend 
             figsize = None, # size of the figure 
             **kwargs) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
        """
            Plots the mean and variance of the runs into a given matplotlib figure and axis.
        """
    
        # Create a figure and axis if nor already given 
        if fig == None and ax == None:
            from src.utils import gen_fig_ax
            fig, ax = gen_fig_ax(keys,figsize=figsize)

        # Create a heading for the figure
        if heading == None:
            #heading = f"Results for the {self.model.plotname}, bs={self.optimizer.bs}, lr={self.optimizer.lr}, neurons={self.model.neurons}"
            heading = f"Results for the {self.model}, bs={self.optimizer.bs}, lr={self.optimizer.lr}"
        fig.suptitle(heading)
        
        keydict = {'lrs':"learning rate", "train_losses": "Training loss", "errors": "error", "norm_diffs": "norm of parameters"}

        for i, key in enumerate(keys):
            label = self.optimizer.plotlabel
            color = self.optimizer.plotcolor
            linestyle = self.optimizer.linestyle
            linewidth = self.optimizer.linewidth
            marker = None 
            if hasattr(self.optimizer, "marker"):
                marker = self.optimizer.marker
            markersize = None 
            if hasattr(self.optimizer,"markersize"):
                markersize=self.optimizer.markersize
            y_label = keydict[key]

            # This line of code is necessary to avoid bugs when we only plot one thing 
            axe = ax[i] if len(keys) > 1 else ax

            axe.set_xlabel("gradient steps")
            axe.set_ylabel(y_label)
            axe.set_yscale(scale)

            if plotmode == "mean":
                #plot the mean and variance 
                mean, var = self.mean_variance()
            
                X_values = mean["step_list"]
                Y_values = mean[key]
                Y_lower = mean[key] - np.sqrt(var[key])
                Y_upper = mean[key] + np.sqrt(var[key])
                
                axe.plot(X_values, Y_values, label = label, color = color, linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
                axe.fill_between(X_values, Y_lower, Y_upper, color = color, alpha = shade_for_variance)
            
            elif plotmode == "all":
                for j,res in enumerate(self.data):
                
                    X_values = res["step_list"]   
                    Y_values = res[key]      
                    axe.plot(X_values, Y_values, 
                             label = label if j == 0 else None, # only print a label for one of the runs 
                             color = color, linestyle = linestyle, linewidth = linewidth)
                        
        if legend:
            axe.legend(bbox_to_anchor=bbox, loc=loc, borderaxespad=0,**kwargs)
            plt.tight_layout()
        if grid:
            axe.grid(True)
        if savefig:
            path = TEST_RESTULT_PLOT_PATH + self.filename + ".pdf"
            plt.savefig(path)
        if return_:
            return fig, ax 
        
    def save_to_pickle(self, filename: str = None) -> None:
        """
            this stores the instance in a pickle file
        """
        if filename is None:
            filename = self.filename 
        path = TEST_RESULT_DATA_PATH + filename + ".pkl" 
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Sucessfully pickled the result at {filename}")

    def generate_filename(self) -> str:
        """ 
            generates a filename containing some info about the model and the optimizer and the current time 
        """
        if hasattr(self,"model"):
            modelname = self.model
        else:
            modelname = "Unknown model"
        filename = f"{modelname}_{self.optimizer.plotlabel}_bs={self.optimizer.bs}_lr={self.optimizer.lr}_steps={self.optimizer.nr_steps}"
        from src.utils import append_hash
        filename = append_hash(filename)
        self.filename = filename 
        return filename 
        

class OptimizationResults:
    """
        A class that contains a list of OptimizationResult, coming from one model and multiple optimizers.
    """
    def __init__(self, datalist: list[OptimizationResult] = []) -> None :
        if datalist is None:
            self.datalist = []
        else:
            self.datalist = datalist 

    def plot(self, 
             keys: list[str] = ["errors"], #, "norm_diffs"], #["errors",'train_losses','lrs', "norm_diffs"], # select the data that you want to plot 
             scale: str = "log", # scale in which you want to plot. Usually "log" or "linear"
             heading: str = None, # You can choose a heading. Else, the function will automatically generate one.
             savefig: bool = True, # Choose if you want to save the figure.
             savedata = True,
             plotmode = "mean", # Choose if you only want to plot the "mean" of several runs or "all"
             figsize = None, # the figure size for generating the figure axis
             shade_for_variance: float = 0.0, # do you want to shade the area above and below the line in the plot for multiple runs? enter number between 0.0 (no shade) and 1.0.
             tightlayout:bool = True, # set this to True to avoid the legend overlapping with the plot 
             **kwargs) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
        """
            plots the means and variances of the different optimizers using the OptimizationResult.plot method 
        """
        
        # generate a matplotlib figure and axis 
        from src.utils import gen_fig_ax
        fig, ax = gen_fig_ax(keys, figsize)

        # If the Optimizationresults has a heading, use this one 
        if heading is None and hasattr(self,"heading"):
            heading = self.heading 

        # plot all the different results from the datalist.
        for result in self.datalist:
            fig, ax = result.plot(fig = fig, ax = ax, keys= keys, scale = scale, heading = heading, return_ = True, plotmode = plotmode,shade_for_variance=shade_for_variance,**kwargs)
        if tightlayout:
            plt.tight_layout()

        if savefig:
            if hasattr(self,"filename"):
                filename = self.filename
            else:
                filename = self.generate_filename()
            path = TEST_RESTULT_PLOT_PATH + filename + ".pdf"#".png"
            plt.savefig(path)
        if savedata:
            self.save_to_pickle()
        return fig, ax 

    def append(self, result: OptimizationResult) -> None:
        """ 
            Appends an OptimizationResult to the datalist
        """
        self.datalist.append(result)
    
    def color(self, colormap = None) -> None:
        """
            colors the optimizers for the plot 
            the colormap can give a function i -> i defining the coloring (default identiy)
        """
        if colormap == None:
            colormap = lambda x:x
        for i, data in enumerate(self.datalist):
            data.optimizer.plotcolor = f"C{colormap(i)}"
    
    def is_empty(self) -> bool:
        """
            Returns True if self.datalist is empty 
        """
        if not self.datalist:
            return True
        else:
            return False

    def generate_filename(self):
        """
            generates a filename containing info about the model and current time 
        """
        if hasattr(self,"model") and isinstance(self.model,str):
            filename = self.model
        elif self.is_empty():
            filename = f"empty_list"
        else:
            self.model = self.datalist[0].model 
            filename = self.model
        from src.utils import append_hash
        filename = append_hash(filename)
        self.filename = filename
        return filename

    def save_to_pickle(self, filename: str =None) -> None:
        """
            this stores the instance in a pickle file
        """
        # Choose a filename depending on whether it has been passed as an argument and depending on whether the object has a filename yet
        if filename is None:
            if hasattr(self, "filename"):
                filename = self.filename
            else:
                filename = self.generate_filename()
        
        # Choose a path and save the file 
        path = TEST_RESULT_DATA_PATH + filename + ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Sucessfully pickled the result at {filename}")
    

    def save_to_csv(self, 
                    folder: str = TEST_RESULT_DATACSV_PATH # folder where the results should be saved 
                    ) -> None:
        """ saves the result in a .csv file """
        from src.utils import create_dir # for some reason this is not imported automatically 
        if not hasattr(self, "filename"):
            self.generate_filename()
        parent_dir = folder + self.filename # This is the directory where all the .csv files from this result will be stored
        create_dir(parent_dir)
        for data in self.datalist:
            opt_dir = parent_dir + "/" + data.optimizer.plotlabel # directory where all results for one optimizer will be stored
            create_dir(opt_dir)
            for i, df in enumerate(data.data):
                path = opt_dir + "/run_" + f"{i}" + ".csv"
                df.to_csv(path, index=False)

        

        

def test_optimizers(model: Model, 
                    optimizer_list: list[Optimizer],
                    plot: bool = True, # If True, results are plotted 
                    savedata: bool = True, # If True, results are stored in a pickle file
                    savefig: bool = False, # If True, we save the figure 
                    stop_time: bool = True, # if True, it measures the time and prints it in the console 
                    save_logs: bool = False, # if True, the console output is saved under tests/log/filename.log
                    save_csv:bool = True, # if true, results are also saved as csv files. 
                    *args,
                    **kwargs) -> OptimizationResults:
    """
        tests a list of optimizers on a given model and returns the results as an instance of OptimizationResults
    """
    print(f"Testing model {model.plotname}")
    result = OptimizationResults(datalist=[]) 
    result.model = model.plotname + "_" + str(model.neurons) # creating a model for the result so that we can generate a filename 
    result.generate_filename()
    # Doing the actual optimization 
    num_opt = len(optimizer_list)
    for i, opt in enumerate(optimizer_list):
        print(f"Testing {opt.plotlabel}({i+1}/{num_opt})")
        if stop_time: 
            start_time = time()  # Record the start time
        res = opt.optimize_several(model) # here the actual optimization happens
        
        if savedata: # Save intermediate steps 
            result.save_to_pickle()
        if stop_time: 
            end_time = time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            res.time = elapsed_time # saving the time in the result
            # message to be stored in the log
            message = f"{opt.plotlabel} for {model.plotname}, bs={opt.bs}, n_test={opt.n_test}, steps={opt.nr_steps} took {elapsed_time:.4f} seconds\n"
            if hasattr(res,"testtime"):
                message += f"{opt.plotlabel} for {model.plotname}, bs={opt.bs}, n_test={opt.n_test}, steps={opt.nr_steps} took testtime of {res.testtime:.4f} seconds\n"
            print(message) # print to console 
            if save_logs:
                logfile = TEST_RESULT_LOG_PATH + result.filename + ".log"
                with open(logfile,"a") as f:
                    f.write(message)
        result.append(res) # appending the result 
    if plot:
        result.plot(savefig=savefig, **kwargs)
    if savedata:
        result.save_to_pickle()
    if save_csv:
        result.save_to_csv()
    return result 

    
def load_result_from_file(filename: str) -> OptimizationResults:
    """loads a test result from a file"""
    # removing .pdf and adding .pkl to filename if necessary. 
    if filename.endswith(".pdf"):
        filename = filename.rstrip(".pdf")
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    # path where the results are stored. 
    folder_path = TEST_RESULT_DATA_PATH
    path = folder_path + filename
    with open(path, "rb") as f:
        result = pickle.load(f)
    return result 