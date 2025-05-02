import torch 
import torch.nn as nn
import numpy as np 
import math 

from src.initializers import *
from src.PDEs_ScikitFEM import *
from src.utils import *
from src.models import *

"""
    This file creates quick preconfigured models for testing the different optimizers.
"""

def configured_model(name: str) -> Model:
    """
        Spits out a configured model using standard configuration. 
    """
    if name == "quadratic":
        return QuadraticModel(1,0,10)
    elif name == "heat":
        # Defining the parameters for the Heat equation
        d_i = 10 
        T = 2.
        rho = 1.
        space_bounds = [-1, 1]
        activation = nn.GELU()
        neurons = [d_i, 50,100, 50, 1]

        def phi(x):  # initial value of solution of heat PDE
            return x.square().sum(axis=1, keepdim=True) 

        def u_T(x):  # value of heat PDE solution at final time T
            return x.square().sum(axis=1, keepdim=True) + 2. * rho * T * d_i

        return Heat_PDE_ANN(neurons, phi, space_bounds, T, rho, dev=give_device() , activation= activation, final_u= u_T)
    elif name == "blackscholes":
        d_i = 10
        neurons = [d_i, 200, 300, 200, 1]
        T = 1.
        r, c, K = 0.05, 0.1, 100.
        sigma = torch.linspace(start=1. / d_i, end=.5, steps=d_i)

        def phi(x):  # initial value 
            return torch.tensor(np.exp(-r * T)) * torch.maximum(torch.max(x, dim=-1, keepdim=True)[0] - K, torch.tensor(0.))
        n_test = 10000
        space_bounds = [90, 110]
        mc_samples = 1024
        mc_rounds = 10000

        return BlackScholes_ANN(neurons, phi, space_bounds, T, c, r, sigma, dev=give_device(), activation = nn.GELU(), 
                            mc_samples=mc_samples, mc_rounds=mc_rounds, test_size=n_test).to(dev) 
    elif name == "allencahn":
        
        activation = nn.GELU()
        neurons = [3, 32,64,32, 1]
        space_bounds = [2., 1.]

        def init_func(x):
            return torch.prod(torch.sin(torch.pi * x), dim=0)


        pde_name = 'AC'
        if pde_name == 'AC':
            nonlin = allen_cahn_nonlin
            torch_nonlin = allen_cahn_nonlin
            T = 4.
            alpha = 0.01
            f_0 = init_func

        else:
            raise ValueError(f'{pde_name}_PDE has not been implemented.')

        FP = True
        train_points = 60000
        test_points = 10000

        ann = SemilinHeat_PINN_2d(neurons, f_0, nonlin, alpha, space_bounds, T, activation=activation, torch_nonlin=torch_nonlin,
                                fixed_points=FP, train_points=train_points, test_points=test_points)
        ann.plotname = "Allen Cahn PINN"
        ann.id = ann.plotname
        return ann

    elif name == "deepritz":
        def phi(x):
            return torch.sum(x.square(), dim=1)
        d_i = 10
        width = 32
        depth = 6
        activation = nn.GELU()
        beta = 500.
        f = 2. * d_i
        residual = True

        return HeatRitz_ANN(d_i, width, depth, phi, give_device(), activation, beta=beta, res=residual, f_term=f).to(give_device())
    elif name == "supervised":
        d = 20
        neurons = [d,300,500,100,1]
        sigma2 = 3
        def f(x):
            norm2 = x.square().mean()
            return torch.exp(-norm2 / (2*sigma2))
        space_bounds = [-2,2]
        return Supervised_ANN(neurons,f,space_bounds,dev)
    elif name == "optimalcontrol":
        activation = nn.ReLU()

        d_i = 10
        neurons = [d_i, d_i + 10, d_i + 10, d_i]
        T = 1.
        nt = 50
        xi = torch.zeros([d_i]).to(dev)

        def phi(x):  # terminal value of solution of HJB PDE
            return torch.log(0.5 * (torch.sum(x.square(), dim=-1) + 1.))

        def f(x, v):  # nonlinear term /running cost
            return torch.sum(v.square(), dim=-1)  # Hamilton-Jacobi-Bellman PDE

        def mu(x, v):  # drift term
            return 2. * v

        mc_size = 4000
        
        mc_samples = 10000
        batch_normalize = True  # determine if using batch normalization in hidden ANN layers

        v = torch.zeros([1]).to(dev)

        print(f'Computing reference solution by MC method with {mc_size * mc_samples} samples.')

        for _ in range(mc_samples):
            w = torch.randn([mc_size, d_i]).to(dev)
            v += torch.mean(torch.exp(- phi(xi + math.sqrt(2. * T) * w)))
        v /= mc_samples
        u_0 = - torch.log(v)

        print(f'Reference solution computed. Value: {u_0.cpu().numpy().item()}')

        return ControlNet(neurons, f, phi, mu, T, nt, xi, dev, sigma=math.sqrt(2),
                        activation=activation, u0=u_0, batch_norm=batch_normalize).to(dev)
    
    
    elif name == "BSDEgen": # use this class instead of BSDE
        
        activation = nn.GELU()

        d_i = 10
        neurons = [d_i, d_i + 20, d_i + 20, d_i]
        input_neurons = [d_i, d_i + 20, d_i + 20, 1]
        T = .2
        nt = 20
        def phi(x):  # terminal value of solution of semilinear PDE
            return torch.log(0.5 * (torch.sum(x.square(), dim=-1) + 1.))

        def f(y, v):  # nonlinearity
            return - torch.sum(v.square(), dim=-1)  # Hamilton-Jacobi-Bellman PDE


        mc_size = 2048
        mc_rounds = 400

        batch_normalize = False  # determine if using batch normalization in hidden ANN layers
        initial = 'U'
        space_bounds = [-1., 1.]

        return BSDE_Net_Gen(neurons, input_neurons, f, phi, T, nt, dev, activation,
                        batch_norm=batch_normalize, mc_rounds=mc_rounds, mc_samples=mc_size, initial=initial, space_bounds=space_bounds).to(dev)
    elif name == "BurgersPINN1d":
                
        activation = nn.GELU()
        neurons = [2, 16, 32, 16, 1]
        space_bounds = [2.]
        T = 0.5
        alpha = 0.05
        beta = 1.1
        def init_func(x):
            return 2. * alpha * math.pi * torch.sin(math.pi * x) / (beta + torch.cos(math.pi * x))
        def final_sol(x):
            return 2. * alpha * torch.pi * torch.sin(torch.pi * x) / (beta * torch.exp(torch.tensor(alpha * T * torch.pi ** 2)) + torch.cos(torch.pi * x))
        FP = True
        train_points = 100000
        test_points = 20000

        return  Burgers_PINN_1d(neurons, init_func, alpha, space_bounds, final_sol, T, activation=activation,
                            fixed_points=FP, train_points=train_points, test_points=test_points)
    elif name == "stoppingnet":
        d = 40
        neurons = [d, d + 200, d + 200, 1]
        r = 0.6
        chi = 100. ** (1. / math.sqrt(d))
        K = 95.
        nt = 100
        T = 1.
        beta = torch.unsqueeze(torch.tensor([np.minimum(0.04 * i, 1.6 - 0.04 * i) for i in range(d)], device = dev), 1)
        rho = beta.square().sum() / d
        delta = torch.unsqueeze(torch.tensor([r - rho * (i + 0.5) / d - 0.2 / math.sqrt(d) for i in range(d)],device = dev), 1)
        def g(s, x):
            return torch.exp(-r * s) * torch.maximum(torch.tensor([K], device = dev) - torch.prod(torch.abs(x), dim=1) ** (1. / math.sqrt(d)), torch.tensor([0.], device = dev))
        ts = torch.linspace(0, T, nt + 1, device = dev)
        def x_func(w):
            return torch.exp((r - delta - 0.5 * beta ** 2) * ts + beta * w) * chi
        return StoppingNet(neurons, nt, g, T, x_func, true_output=6.545, activation=nn.GELU())

    elif name == "powerritz":
        d_i = 4
        width = 32
        depth = 4
        ann = LaplacePowerRitz_ANN(d_i=d_i,depth=depth,width=width,dev=dev,beta = 500)
        return ann
    elif name == "darcy":
                
        neurons = [2, 32, 64, 32, 1]

        def conductivity(x):
            return 4 + x[:, 0] + 2. * x[:, 1]
        domain = 'S'
        boundary_loss_factor = 1.
        ann = DarcyPINN_2d(neurons, conductivity, 1., dev, domain=domain, nr_refine=7, b_factor=boundary_loss_factor,
                        activation=nn.GELU()).to(dev)
        return ann
    elif name == "polyreg": 
        deg = 25
        n = 50000
        sigma = 0.2

        def P(x): # target function
            return torch.sin(math.pi * x)

        ann = PolyReg1d(deg, P, n, dev, sigma)
        return ann
    else:
        raise ValueError("No model known with this name")
