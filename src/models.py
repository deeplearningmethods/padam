"""
This file contains all the trainable ANN models for the simulations
"""


import torch 
import torch.nn as nn
import numpy as np 
import math 
from typing import Callable 
from tqdm import tqdm
from src.initializers import *
from src.PDEs_ScikitFEM import *
from src.utils import *


class Model(nn.Module):
    """
        A class that serves as an abstract base class for ANN models 
    """
    def __init__(self) -> None:
        super().__init__()
        self.id = None # deprecated, generates an id
        self.plotname = None # a string that is used in plots later
        self.layers = nn.ModuleList() # the layers
        self.initializer = None # the sampler used for sampling the data
        self.neurons = None # a list containing the numbers of neurons of each layer

    # The forward pass of the network 
    def forward(self, data: torch.Tensor) -> torch.Tensor: 
        for fc in self.layers:
            data = fc(data)
            return data
        
    # the training loss of the network
    def loss(self, data: torch.Tensor) -> torch.Tensor:
        pass

    # the test loss (in most cases, relative L2-error)
    def test_loss(self, data: torch.Tensor) -> torch.Tensor:
        pass

    # a function that re-initializes the weights for multiple training runs
    def initialize_weights(self) -> None:
        def initializing(m: nn.Parameter) -> None:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Parameter):
                nn.init.normal_(m, mean=0.0, std=1.0)
        self.apply(initializing)
        return 


class Heat_PDE_ANN(Model):
    """
        This model solves the Heat PDE using the deep Kolmogorov method
    """
    def __init__(self, 
                 neurons: list[int], 
                 phi: Callable, 
                 space_bounds: list[float], 
                 T: float, 
                 rho: float, 
                 dev : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 activation: torch.nn = nn.ReLU(), 
                 final_u: Callable  = None) -> None:

        super().__init__()
        # default attributes
        self.id = "heat"
        self.plotname = "Heat Model"

        # custom attributes
        self.neurons = neurons
        self.dims = neurons
        self.phi = phi
        self.space_bounds = space_bounds
        self.T = T
        self.rho = rho
        self.final_u = final_u

        depth = len(neurons) - 1
        for i in range(depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))
        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
       
    def loss(self, data) -> torch.Tensor:
        W = torch.randn_like(data)
        return (self.phi(math.sqrt(2 * self.rho * self.T) * W + data) - self.forward(data)).square().mean()
    
    def forward(self, x): # for some reason, I should include this here again 
        for fc in self.layers:
            x = fc(x)
        return x

    def test_loss(self, data) -> None:
        output = self.forward(data)
        u_T = self.final_u(data)
        return ((u_T - output)).square().mean().sqrt() / u_T.square().mean().sqrt()



class BlackScholes_ANN(Model):
    def __init__(self, 
                 neurons, 
                 phi, 
                 space_bounds, 
                 T, 
                 c, 
                 r, 
                 sigma, 
                 dev, 
                 activation=nn.ReLU(), 
                 final_u=None, 
                 mc_samples=1024, 
                 test_size=10000, 
                 mc_rounds=100):
        super().__init__()
        # default attributes
        self.id = "blackscholes"
        self.plotname = "Black Scholes Model"

        # custom attributes
        self.layers = nn.ModuleList()
        self.neurons = neurons 
        self.dims = neurons
        self.depth = len(neurons) - 1
        self.layers.append(nn.BatchNorm1d(neurons[0]).to(dev))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.phi = phi
        self.x_test = self.initializer.sample(test_size).to(dev)
        self.dev = dev 
        self.mu = r - c
        self.r = r
        self.sigma = sigma.to(dev)
        self.mc_samples = mc_samples
        self.T = T
        self.space_bounds = space_bounds
        self.final_u = final_u

        self.done_steps = 0

        self.losses = []
        self.lr_list = []

        if self.final_u is None:  # approximate true solution by MC
            print(f'Computing reference sol by MC with {mc_rounds} rounds and {mc_samples} samples.')
            u_ref = torch.zeros([test_size, neurons[-1]],device = self.dev)
            for i in tqdm(range(mc_rounds)):
                x = torch.stack([self.x_test for _ in range(self.mc_samples)])
                w = torch.randn_like(x, device = dev)
                u = self.phi(self.x_test * torch.exp((self.mu - 0.5 * self.sigma ** 2) * self.T + self.sigma * torch.tensor(math.sqrt(self.T),device = dev) * w))
                u = torch.mean(u, dim=0)
                u_ref += u
            self.u_test = u_ref / mc_rounds
            print(f'Reference sol computed, shape: {self.u_test.shape}.')
        else:
            self.u_test = self.final_u(self.x_test)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        W = torch.randn_like(data, device = dev)
        X = data * torch.exp((self.mu - 0.5 * self.sigma ** 2) * self.T + self.sigma * math.sqrt(self.T) * W)
        return (self.phi(X).to(dev) - self.forward(data).to(dev)).square().mean()

    def test_loss(self, data):
        output = self.forward(self.x_test)

        return ((self.u_test - output) ).square().mean().sqrt()/ self.u_test.square().mean().sqrt()
    


class BiasLayer(torch.nn.Module):
    # A layer of a d-dimensional bias which is initialized by a standard normal distribution
    def __init__(self,d):
        super().__init__()
        bias_value = torch.randn(d)
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return 0*x + self.bias_layer # return one copy of the bias for every input dimension of x 


class QuadraticModel(Model):
    """
        This modules solves the quadratic minimization problem l(x,theta) = || theta - x ||^2 where theta is initialized as a standard normal distr
    """

    def __init__(self, variance, mean, d):
        super().__init__()
        # default attributes
        self.id = "quadratic_minimization"
        self.plotname = f"Quadratic Minimization Problem, d={d}, var={variance}, mean={mean}"

        # custom attributes
        self.neurons = [d]
        self.variance = variance # variance of the data points that are sampled 
        self.mean = mean # mean of the data points that are sampled
        self.d = d # dimension
        self.layers = nn.ModuleList()
        self.layers.append(BiasLayer(d))
        self.initializer = NormalValueSampler(d, mean = mean, variance = variance)

    def initialize_weights(self, factor = 1.): # re-initialize the bias 
        for layer in self.layers:
            bias_value = torch.ones_like(torch.randn(self.d), device=dev)
            bias_value += torch.randn_like(bias_value)*0.01
            layer.bias_layer = torch.nn.Parameter(bias_value*factor)

    def forward(self,x):
        for f in self.layers:
            x = f(x)
        return x 
    
    def loss(self,x):
        return (self.forward(x) - x).square().mean()
    
    
    def test_loss(self,x):
        """
            The test loss is just the deviation from the mean squared 
        """
        return (self.forward(x)-self.mean).square().mean()
    

class SemilinHeat_PINN_2d(Model):
    """
        Neural network to solve semilinear heat equation du/dt = alpha * Laplace(u) + nonlin(u) on rectangle [0, a] x [0, b]
        with either Dirichlet or periodic boundary conditions using PINN method.
    """

    def __init__(self, neurons, f, nonlin, alpha, space_bounds, T=1., boundary='D', test_discr=100, test_timesteps=500,
                 activation=nn.Tanh(),
                 nonlin_name=None, torch_nonlin=None, fixed_points=False, train_points=1, test_points=1,dev : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.neurons = neurons
        # default attributes
        self.id = "semilinheatpinn"
        self.plotname = f"SemilinearHeatPINN"

        assert neurons[0] == 3
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]).to(dev))
        self.f = f
        self.T = T
        self.alpha = alpha
        self.space_bounds = space_bounds
        self.spacetime_bounds = space_bounds + [T]
        self.boundary = boundary

        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        if torch_nonlin is None:
            self.torch_nonlin = nonlin
        else:
            self.torch_nonlin = torch_nonlin

        self.base = FEM_Basis(space_bounds, 1, test_discr)
        self.ref_method = ReferenceMethod(Second_order_linear_implicit_RK_FEM, self.alpha, torchfunc_to_numpyfunc(self.nonlin),
                                          self.nonlin_name, None, (0.5, 0.5), 'LIRK2')
        self.pde = self.ref_method.create_ode(self.base)
        self.init_values = self.base.project_cont_function(torchfunc_to_numpyfunc(f))


        self.final_sol = self.ref_method.compute_sol(T, self.init_values, test_timesteps, self.pde, self.base)
        self.u_t_fem = self.base.basis.interpolator(self.final_sol)

        initializer = RectangleValueSampler(3, self.spacetime_bounds,)
        if fixed_points:  # sample points once and draw batches from these for training (PINN-style), not used at the moment 
            self.train_data = initializer.sample(train_points)
            self.initializer = DrawBatch(self.train_data)
            self.test_data = initializer.sample(test_points)
            self.test_sampler = DrawBatch(self.test_data)
        else:
            self.initializer = initializer  # sample new points in each training step (Galerkin-style)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):


        x =data[:, 0:2]
        t =data[:, 2:3]
        f0 =(self.f(data[:, 0:2].t())).t()
        x.requires_grad_()
        t.requires_grad_()
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x0 = torch.cat((x, torch.zeros_like(t)), 1)
        u0 = torch.squeeze(self.forward(x0))
        initial_loss = (u0 - f0).square().mean()

        u = self.forward(torch.cat((x1, x2, t), 1))
        u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0]
        u_xx1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]
        u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
        u_xx2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        loss = (self.alpha * (u_xx1 + u_xx2) + self.torch_nonlin(u) - u_t).square().mean()
        xa0 = torch.cat((torch.zeros_like(x1), x2, t), 1)
        xa1 = torch.cat((self.space_bounds[0] * torch.ones_like(x1), x2, t), 1)
        xb0 = torch.cat((x1, torch.zeros_like(x2), t), 1)
        xb1 = torch.cat((x1, self.space_bounds[1] * torch.ones_like(x2), t), 1)
        if self.boundary == 'D':
            boundary_loss = (self.forward(xa0)).square().mean() + (self.forward(xa1)).square().mean() \
                            + (self.forward(xb0)).square().mean() + (self.forward(xb1)).square().mean()
        elif self.boundary == 'P':
            boundary_loss = (self.forward(xa0) - self.forward(xa1)).square().mean() \
                            + (self.forward(xb0) - self.forward(xb1)).square().mean()
        else:
            raise ValueError('Boundary condition must be either "D" (Dirichlet) or "P" (Periodic)')
        loss_value = loss + boundary_loss + 2. * initial_loss       
        return loss_value

    def test_loss(self, x):
        x = x[:, 0:2]
        y_t_fem = self.u_t_fem(x.t().cpu().detach().numpy())
        x_t = torch.cat((x, self.T * torch.ones_like(x[:, 0:1])), 1)
        y_t_net = torch.squeeze(self.forward(x_t)).cpu().detach().numpy()
        l2_err = torch.tensor(y_t_fem - y_t_net).square().mean()
        ref = torch.tensor(y_t_fem).square().mean()
        return (l2_err / ref).sqrt()
    

class HeatRitz_ANN(Model):
    def __init__(self, d_i, width, depth, phi, dev, activation=nn.ReLU(), f_term=0., lr=0.0001, res=False, beta=1., dom='C'):
        super().__init__()
        # default attributes
        self.id = "heatritz"
        self.plotname = "Deep Ritz Heat Model"

        # custom attributes
        self.layers = nn.ModuleList()
        self.width = width
        self.d_i = d_i
        self.activation = activation
        self.depth = depth
        self.layers.append(nn.Linear(self.d_i, self.width))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(self.width, self.width))
        self.layers.append(nn.Linear(self.width, 1))

        if dom == 'C':
            self.initializer = CubeSampler(d_i, dev)
        elif dom == 'B':
            self.initializer = BallSampler(d_i, dev)
        else:
            raise ValueError('Domain must be either "C" (cube) or "B" (ball)')
        self.phi = phi

        self.done_steps = 0

        self.losses = []
        self.lr_list = []
        self.lr = lr
        self.res = res
        self.beta = beta
        self.f = f_term

    def forward(self, x):
        for fc in range(self.depth + 1):
            x_temp = self.layers[fc](x)
            if fc < self.depth - 1:
                x_temp = self.activation(x_temp)
                if self.res and fc > 0:
                    x_temp += x
            x = x_temp
        return x

    def loss(self, data):
        x_i = data[:, 0:self.d_i]
        x_b = data[:, self.d_i:2 * self.d_i]
        x_i.requires_grad_()

        y_i = self.forward(x_i)
        y_b = self.forward(x_b)

        dfdx = torch.autograd.grad(y_i, x_i, grad_outputs=torch.ones_like(y_i), create_graph=True)[0]

        ssum = 0.5 * torch.sum(dfdx.square(), dim=1) + self.f * y_i
        l_i = ssum.mean()
        l_b = (self.phi(x_b) - y_b.squeeze()).square().mean()

        return l_i + self.beta * l_b

    def test_loss(self, data):
        x = data[:, 0:self.d_i]
        y = self.forward(x).squeeze()
        y_true = self.phi(x)
        l2_err = (y - y_true).square().mean()
        ref = y_true.square().mean()
        return (l2_err / ref).sqrt()
    

    
class PolyReg1d(Model):
    """Modeling polynomial regression in one dimension, which leads to a convex problem (random feature model)."""

    def __init__(self, deg, P, n_i, dev, sigma=1., space_bounds=(-1., 1.)):
        super().__init__()
        self.id = "polyreg"
        self.plotname = "1d Polynomial regression"
        self.neurons = []
        self.deg = deg
        self.space_bounds = space_bounds
        self.P = P
        self.x_i = initial_values_sampler_uniform(n_i, 1, space_bounds)
        print(f'Input data shape:', self.x_i.shape)

        self.A = torch.cat([self.x_i ** k for k in range(self.deg + 1)], 1)
        print(f'Design matrix shape:', self.A.shape)

        self.params = torch.nn.Parameter(torch.randn(self.deg + 1))
        self.sigma = sigma
        self.initializer = UniformValueSampler(1, space_bounds, dev)

    def loss(self, data):
        y = self.P(data) + math.sqrt(self.sigma) * torch.randn_like(data)
        A = torch.cat([data ** k for k in range(self.deg + 1)], 1)  # design matrix (monomials)
        d = torch.matmul(A, self.params) - torch.squeeze(y)
        return d.square().mean()

    def test_loss(self, data):
        y = self.P(self.x_i)
        d = torch.matmul(self.A, self.params) - torch.squeeze(y)
        return d.square().mean().sqrt()
    



    
class Supervised_ANN(Model):
    """
    Approximates deterministic target function f by neural network.
    neurons: List/Tuple specifying the layer dimensions.
    """

    def __init__(self, neurons, f, space_bounds, dev, activation=nn.ReLU(), sigma=0.):
        super().__init__()
        self.id = "supervised"
        self.plotname = "supervised problem"
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.neurons = neurons 
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.target_fn = f

        self.done_steps = 0
        self.sigma = sigma

        self.losses = []
        self.lr_list = []

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, x):
        u = self.target_fn(x)
        y = u + math.sqrt(self.sigma) * torch.randn_like(u)  # output contains random noise
        output = self.forward(x)
        l = (y - output).square().mean()
        return l

    def test_loss(self, x):
        y = self.forward(x)
        l = (y - self.target_fn(x)).square().mean().sqrt()
        return l
    

class ControlNet(Model):
    """Approximates solution of optimal control problems.

    Args:
    neurons: list/tuple of neural network layer dimensions
    L: nonlinearity/running cost (function)
    phi: terminal cost
    mu: drift term, depending on x-value and control
    T: final time
    nt: number of Euler time steps
    xi: fixed space point where cost function is computed
    dev: device on which model tensors are generated
    sigma: diffusion coefficient
    activation: activation function
    u0: value of reference solution at point xi
    batch_norm: bool that determines if batch normalization is used
    """
    def __init__(self, neurons, L, phi, mu, T, nt, xi, dev, sigma=1., activation=nn.ReLU(), u0=None, batch_norm=True):
        super().__init__()
        self.id = "controlnet"
        self.plotname = "Optimal Control Problem"
        self.neurons =neurons 
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(nt - 1)])
        self.dims = neurons
        self.d_i = neurons[0]
        self.depth = len(neurons) - 1
        for k in range(nt - 1):
            for i in range(self.depth - 1):
                self.layers[k].append(nn.Linear(neurons[i], neurons[i + 1]))
                self.layers[k].append(activation)
                if batch_norm:
                    self.layers[k].append(nn.BatchNorm1d(neurons[i + 1]))
            self.layers[k].append(nn.Linear(neurons[-2], neurons[-1]))
        self.dev = dev
        self.initializer = BrownianPathSampler(neurons[0], nt, dev)
        self.phi = phi  # initial condition/terminal cost
        self.f = L  # nonlinearity/running cost
        self.mu = mu  # drift term
        self.sigma = sigma  # diffusion coefficient (scalar)
        self.V0 = nn.Parameter(2. * torch.rand([self.d_i]) - 1.)

        self.T = T
        self.nt = nt
        self.dt = T / nt
        self.xi = xi.to(dev)

        self.done_steps = 0

        self.u_0 = u0  # reference solution value

    def loss(self, w):
        w = w * math.sqrt(self.dt)
        l = self.f(self.xi, self.V0) * self.dt * torch.ones([w.shape[0]]).to(self.dev)
        X = self.xi + self.mu(self.xi, self.V0) * self.dt + self.sigma * w[:, :, 0]
        for i in range(self.nt - 1):
            V = X
            for fc in self.layers[i]:
                V = fc(V)
            l += self.f(X, V) * self.dt
            X = X + self.mu(X, V) * self.dt + self.sigma * w[:, :, i + 1]

        l += self.phi(X)
        return l.mean()

    def test_loss(self, w):
        w = w * math.sqrt(self.dt)
        l = self.f(self.xi, self.V0) * torch.ones([w.shape[0]]).to(self.dev)
        X = self.xi + self.mu(self.xi, self.V0) * self.dt + self.sigma * w[:, :, 0]

        for i in range(self.nt - 1):
            V = X
            for fc in self.layers[i]:
                V = fc(V)
            l += self.f(X, V) * self.dt
            X = X + self.mu(X, V) * self.dt + self.sigma * w[:, :, i + 1]

        l += self.phi(X)
        return ((l.mean() - self.u_0) / self.u_0).abs()



def HJB_compute_ref(x_test, mc_samples, mc_rounds, phi, T, sigma=math.sqrt(2)):
    # compute solution of HJB PDE via Monte-Carlo method and Cole-Hopf transform
    u_ref = torch.zeros([x_test.shape[0]]).to(x_test.device)
    for i in range(mc_rounds):
        x = torch.stack([x_test for _ in range(mc_samples)])
        w = torch.randn_like(x)
        u = torch.exp(- phi(x_test + sigma * math.sqrt(T) * w))
        u = torch.mean(u, dim=0)
        u_ref += u
    u_test = - torch.log(u_ref / mc_rounds)
    return u_test


class BSDE_Net_Gen(Model):
    """implements the deep BSDE method by https://arxiv.org/abs/1706.04702"""
    def __init__(self, neurons, input_neurons, f, g, T, nt, dev, activation=nn.ReLU(), lr=0.0001,
                 batch_norm=True, test_size=1024, mc_samples=10000, mc_rounds=100, space_bounds=None, initial='N'):
        super().__init__()
        self.id = "BSDEgen"
        self.plotname = "BSDEgen"
        self.neurons = neurons 
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(nt)])
        self.network = nn.ModuleList()
        self.dims = neurons
        self.d_i = neurons[0]
        self.depth = len(neurons) - 1
        for i in range(len(input_neurons) - 2):
            self.network.append(nn.Linear(input_neurons[i], input_neurons[i + 1]))
            self.network.append(activation)
            if batch_norm:
                self.network.append(nn.BatchNorm1d(input_neurons[i + 1]))
        self.network.append(nn.Linear(input_neurons[-2], input_neurons[-1]))
        for k in range(nt):
            for i in range(self.depth - 1):
                self.layers[k].append(nn.Linear(neurons[i], neurons[i + 1]))
                self.layers[k].append(activation)
                if batch_norm:
                    self.layers[k].append(nn.BatchNorm1d(neurons[i + 1]))
            self.layers[k].append(nn.Linear(neurons[-2], neurons[-1]))

        if initial == 'N':
            self.initializer = NormalValueSampler(self.d_i, dev)
        elif initial == 'U' and space_bounds is not None:
            self.initializer = UniformValueSampler(self.d_i, space_bounds, dev)
        else:
            raise ValueError('Initializer must be N or U, and in the latter case space-bounds cannot be None.')
        self.phi = g  # initial condition
        self.f = f  # nonlinearity

        self.T = T
        self.nt = nt
        self.dt = T / nt

        self.done_steps = 0

        self.losses = []
        self.lr_list = []
        self.lr = lr

        self.x_test = self.initializer.sample(test_size)

        print(f'Computing reference sol by MC with {mc_rounds} rounds and {mc_samples} samples.')
        self.u_test = HJB_compute_ref(self.x_test, mc_samples, mc_rounds, self.phi, T)
        print(f'Reference sol computed, shape: {self.u_test.shape}.')

    def forward(self, x):
        for fc in self.network:
            x = fc(x)
        return x

    def loss(self, data):
        Y0 = self.forward(data).squeeze()
        V0 = data
        for fc in self.layers[0]:
            V0 = fc(V0)
        w = torch.randn_like(data) * math.sqrt(2. * self.dt)
        Y = Y0 - self.f(Y0, V0) * self.dt + torch.sum(V0 * w, dim=-1)
        X = data + w
        for i in range(1, self.nt):
            V = X
            w = torch.randn_like(data) * math.sqrt(2. * self.dt)
            for fc in self.layers[i]:
                V = fc(V)

            Y = Y - self.f(Y, V) * self.dt + torch.sum(V * w, dim=-1)
            X = X + w

        return (Y.squeeze() - self.phi(X)).square().mean()

    def test_loss(self, x):
        output = self.forward(self.x_test).squeeze()

        return ((self.u_test - output) / self.u_test).square().mean().sqrt()
    

class Burgers_PINN_1d(Model):
    # ANN to solve equation du/dt = alpha * Laplace(u) - u * du / dx on interval (0, a) with 0 boundary conditions
    def __init__(self, neurons, f, alpha, space_bounds, final_sol, T=1., activation=nn.Tanh(), fixed_points=False,
                 train_points=1, test_points=1):
        super().__init__()
        self.id= "BurgersPINN1d"
        self.plotname = "BurgersPINN1d"
        self.neurons = neurons
        self.layers = nn.ModuleList()
        self.dims = neurons
        assert neurons[0] == 2
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.f = f
        self.alpha = alpha
        self.T = T
        self.space_bounds = space_bounds
        self.spacetime_bounds = space_bounds + [T]
        self.final_sol = final_sol

        initializer = RectangleValueSampler(2, self.spacetime_bounds)

        if fixed_points:  # sample points once and draw batches from these for training (PINN-style)
            self.train_data = initializer.sample(train_points)
            self.initializer = DrawBatch(self.train_data)
            self.test_data = initializer.sample(test_points)
            self.test_sampler = DrawBatch(self.test_data)
        else:
            self.initializer = initializer  # sample new points in each training step (Galerkin-style)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        x = torch.Tensor(data[:, 0:1])
        t = torch.Tensor(data[:, 1:2])
        x.requires_grad_()
        t.requires_grad_()

        u = self.forward(torch.cat((x, t), 1))
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        loss = (self.alpha * u_xx - u * u_x - u_t).square().mean()

        xt0 = torch.cat((torch.zeros_like(t), t), 1)
        xt1 = torch.cat((self.space_bounds[0] * torch.ones_like(t), t), 1)

        boundary_loss = (self.forward(xt0)).square().mean() + (self.forward(xt1)).square().mean()

        x0 = torch.cat((x, torch.zeros_like(x)), 1)

        initial_loss = (self.forward(x0) - self.f(x)).square().mean()

        loss_value = loss + boundary_loss + initial_loss
        return loss_value

    def test_loss(self, data):
        x = data[:, 0:1]
        y_t_exact = self.final_sol(x.t())
        x_t = torch.cat((x, self.T * torch.ones_like(x)), 1)
        y_t_net = torch.squeeze(self.forward(x_t))
        l2_err = (y_t_exact - y_t_net).square().mean()
        ref = y_t_exact.square().mean()

        return (l2_err / ref).sqrt()



class StoppingNet(Model):
    def __init__(self, neurons, nt, g, T, x_func, activation=nn.GELU(), batch_norm=True, true_output=None,dev : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        
        # default attributes
        self.id = "stoppingnet"
        self.plotname = "Stopping Net"
        self.dev=dev

        # custom attributes
        self.neurons = neurons
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(nt + 1)])
        self.dims = neurons
        self.d_i = neurons[0]
        self.depth = len(neurons) - 1
        self.nt = nt
        for k in range(nt + 1):
            if batch_norm:
                self.layers[k].append(nn.BatchNorm1d(self.d_i).to(dev))
            for i in range(self.depth - 1):
                self.layers[k].append(nn.Linear(neurons[i], neurons[i + 1]).to(dev))
                self.layers[k].append(activation)
            self.layers[k].append(nn.Linear(neurons[-2], neurons[-1]).to(dev))
            self.layers[k].append(nn.Sigmoid())

        self.g = g
        self.ts = torch.linspace(0, T, self.nt + 1)
        self.initializer = BrownianMotionSampler(self.d_i, self.ts)
        self.true_output = true_output
        self.x_func = x_func

    def forward(self, x, index):
        for fc in self.layers[index]:
            x = fc(x)
        x = torch.maximum(x, torch.tensor([1 + index - self.nt], device = self.dev))
        return x

    def loss(self, w):
        x = self.x_func(w)
        u = torch.squeeze(self.forward(x[:, :, 0], 0))
        U_list = [u.clone()]
        u_sum = u.clone()
        gg = self.g(self.ts[0], x[:, :, 0])
        phi = u * gg
        for k in range(self.nt):
            u = torch.squeeze(self.forward(x[:, :, k + 1], k + 1))
            u_mult = u * (1. - u_sum)
            U_list.append(u_mult)
            u_sum += u_mult
            phi += u_mult * self.g(self.ts[k + 1], x[:, :, k + 1])

        return - phi.mean()

    def test_loss(self, w):
        x = self.x_func(w)
        u = torch.squeeze(self.forward(x[:, :, 0], 0))
        U_list = [u.clone()]
        u_sum = u.clone()

        for k in range(self.nt):
            u = torch.squeeze(self.forward(x[:, :, k + 1], k + 1))
            u_mult = u * (1. - u_sum)
            U_list.append(u_mult)
            u_sum += u_mult

        u_stack = torch.stack(U_list, dim=-1)
        sum = u_stack.cumsum(dim=-1)

        comp = torch.tensor(sum + u_stack > 1.).to(int)
        tau = torch.argmax(comp, dim=1)
        ts = self.ts[tau]
        tt = tau.repeat(self.d_i, 1).transpose(0, 1).unsqueeze(dim=-1)
        xtau = torch.gather(x, dim=2, index=tt)
        g_tau = self.g(ts, xtau.squeeze(dim=-1))

        return (g_tau.mean() - self.true_output).abs() / self.true_output




class LaplacePowerRitz_ANN(Model):
    """Solves p-Laplace equation -div(|nabla u|^(p-2) nabla u) = f in d-dimensional unit sphere."""
    def __init__(self, d_i, width, depth, dev, exponent=2., activation=nn.GELU(), f_term=1., res=False, beta=1.):
        super().__init__()
        self.id = "LaplacePowerRitz"
        self.plotname = "LaplacePowerRitz"
        self.neurons = [d_i] + [width for _ in range(depth)] + [1]
        self.layers = nn.ModuleList()

        self.layers = nn.ModuleList()
        self.width = width
        self.d_i = d_i
        self.activation = activation
        self.depth = depth
        self.layers.append(nn.Linear(self.d_i, self.width))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(self.width, self.width))
        self.layers.append(nn.Linear(self.width, 1))

        self.initializer = BallSampler(d_i, dev)
        self.p = exponent

        self.c_pd = (self.p - 1) / self.p * (1. / self.d_i) ** (1. / (self.p - 1))

        self.res = res
        self.beta = beta
        self.f = f_term
        self.dev = dev

    def forward(self, x):
        for fc in range(self.depth + 1):
            x_temp = self.layers[fc](x)
            if fc < self.depth - 1:
                x_temp = self.activation(x_temp)
                if self.res and fc > 0:
                    x_temp += x
            x = x_temp
        return x

    def exact_sol(self, data):
        return self.f * self.c_pd * (1. - (data.square().sum(dim=-1)) ** (0.5 * self.p / (self.p - 1.)))

    def loss(self, data):
        x_i = data[:, 0:self.d_i]
        x_b = data[:, self.d_i:2*self.d_i]
        x_i.requires_grad_()

        y_i = self.forward(x_i)
        y_b = self.forward(x_b)

        dfdx = torch.autograd.grad(y_i, x_i, grad_outputs=torch.ones_like(y_i), create_graph=True)[0]

        ssum = 1. / self.p * torch.sum(dfdx.square(), dim=1) ** (0.5 * self.p) - self.f * y_i
        l_i = ssum.mean()
        l_b = y_b.squeeze().square().mean()

        return l_i + self.beta * l_b

    def test_loss(self, data):
        x = data[:, 0:self.d_i]
        y = self.forward(x).squeeze()
        y_true = self.exact_sol(x)
        l2_err = (y - y_true).square().mean()
        ref = y_true.square().mean()
        return (l2_err / ref).sqrt()




def numpyfunc_from_torch(g, transpose=False):
    def f(x):
        x = torch.tensor(x)
        if transpose:
            x = x.transpose(0, 1)
        gx = g(x).cpu().detach().numpy()
        return gx

    return f

class DarcyPINN_2d(Model):
    """PINN to solve stationary PDE div(K grad u) = g with zero boundary conditions,
    where K (conductivity) fixed function,
    rhs scalar value,
    domain either S (square) or D (disk)"""

    def __init__(self, neurons, conductivity, rhs, dev, activation=nn.Tanh(), domain='S', b_factor=1., nr_refine=4):
        super().__init__()
        self.id="Darcy PINN"
        self.plotname = "Darcy PINN"
        self.neurons = neurons
        self.layers = nn.ModuleList()
        self.dims = neurons
        assert neurons[0] == 2
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.K = conductivity
        self.g = rhs
        self.b_factor = b_factor

        self.dev = dev

        self.base = DomainBasis(domain, nr_refine)
        self.u_fem = compute_darcy_sol(self.base, numpyfunc_from_torch(self.K, transpose=True), self.g)
        self.fem_sol = self.base.basis.interpolator(self.u_fem)

        self.model_name = f'{activation}-PINN, arch. {neurons}, Darcy on {domain}-Domain'

        if domain == 'S':
            self.initializer = CubeSampler(2, dev)
        elif domain == 'D':
            self.initializer = BallSampler(2, dev)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        xi = data[:, :2].to(self.dev)
        xb = data[:, 2:].to(self.dev)
        xi.requires_grad_()
        xb.requires_grad_()

        x1 = xi[:, 0:1]
        x2 = xi[:, 1:2]

        u = self.forward(torch.cat([x1, x2], 1))
        u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0].squeeze()
        u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0].squeeze()

        k_value = self.K(torch.cat([x1, x2], 1))

        u_xx1 = torch.autograd.grad(k_value * u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]
        u_xx2 = torch.autograd.grad(k_value * u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]


        loss = ((u_xx1 + u_xx2) + self.g).square().mean()

        ub = self.forward(xb)

        boundary_loss = ub.square().mean()
        loss_value = loss + self.b_factor * boundary_loss
        return loss_value

    def test_loss(self, data):
        while True:
            x = data[:, :2].to(self.dev)
            try:
                u_fem = torch.tensor(self.fem_sol(np.transpose(x.detach().cpu().numpy()))).to(self.dev)
                break
            except ValueError as e:
                print(f'Error {e} occurred, resampling...')
                data = self.initializer.sample(x.shape[0])

        u = self.forward(x).squeeze().detach()


        l2_err = (u - u_fem).square().mean()
        ref = u_fem.square().mean()

        return (l2_err / ref).sqrt()


class PolyReg1d(Model):
    """Modeling polynomial regression in one dimension, which leads to a convex problem (random feature model)."""

    def __init__(self, deg, P, n_i, dev, sigma=1., space_bounds=(-1., 1.)):
        super().__init__()
        self.id = "Polynomial Regression"
        self.plotname = "Polynomial Regression"
        self.deg = deg
        self.neurons=[deg]
        self.space_bounds = space_bounds
        self.P = P
        self.x_i = initial_values_sampler_uniform(n_i, 1, space_bounds)
        print(f'Input data shape:', self.x_i.shape)

        self.A = torch.cat([self.x_i ** k for k in range(self.deg + 1)], 1)
        print(f'Design matrix shape:', self.A.shape)

        self.params = torch.nn.Parameter(torch.randn(self.deg + 1))
        self.sigma = sigma
        self.initializer = UniformValueSampler(1, space_bounds, dev)

        self.model_name = f'Degree {deg} polynomial regression'

    def loss(self, data):
        y = self.P(data) + math.sqrt(self.sigma) * torch.randn_like(data)
        A = torch.cat([data ** k for k in range(self.deg + 1)], 1)  # design matrix (monomials)
        d = torch.matmul(A, self.params) - torch.squeeze(y)
        return d.square().mean()

    def test_loss(self, data):
        y = self.P(self.x_i)
        d = torch.matmul(self.A, self.params) - torch.squeeze(y)
        return d.square().mean().sqrt()
    



