import torch 
import torch.nn as nn
import numpy as np 
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def initial_values_sampler_uniform(batch_size, space_dim, space_bounds):
    a, b = space_bounds
    return (b - a) * torch.rand([batch_size, space_dim], device = dev) + a
def initial_values_sampler_gaussian(batch_size, space_dim):
    return torch.randn([batch_size, space_dim], device = dev)
def initial_values_sampler_rectangular(batch_size, space_dim, space_bounds,dev = dev ):
    random_tensor = space_bounds *torch.rand((batch_size, space_dim),device = dev)
    return random_tensor
def sample_from_boundary(dim, bs):
    data = 2. * torch.rand(bs, dim, device = dev) - 1.
    norm = torch.norm(data, float('inf'), dim=1, keepdim=True)
    if torch.min(norm) == 0:
        return sample_from_boundary(dim, bs)
    else:
        array = data / norm
        return array
def sample_from_disk(dim, bs):
    """
    dim -- space dimension;
    bs -- number of samples.
    """
    array = torch.randn(bs, dim, device = dev)
    norm = torch.norm(array, 2, dim=1, keepdim=True)
    if torch.min(norm) == 0:
        return sample_from_disk(dim, bs)
    else:
        data = array / norm
        radius = torch.rand(bs, 1, device = dev) ** (1./dim)
        data = data * radius

        return data
def sample_from_sphere(dim, bs):
    array = torch.randn(bs, dim, device = dev)
    norm = torch.norm(array, 2, dim=1, keepdim=True)
    if torch.min(norm) == 0:
        return sample_from_sphere(dim, bs)
    else:
        data = array / norm
        return data
class RectangleValueSampler:
    def __init__(self, space_dim, space_bounds,device = dev):
        self.space_dim = space_dim
        self.device = device 
        assert len(space_bounds) == space_dim
        self.space_bounds = torch.tensor(space_bounds, device=dev)

    def sample(self, batch_size):
        values = initial_values_sampler_rectangular(batch_size, self.space_dim, self.space_bounds,dev=self.device)
        
        return values
class UniformValueSampler:
    def __init__(self, space_dim, space_bounds, dev):
        self.space_dim = space_dim

        assert len(space_bounds) == 2
        self.space_bounds = space_bounds
        self.dev = dev

    def sample(self, batch_size):
        values = initial_values_sampler_uniform(batch_size, self.space_dim, self.space_bounds)
        return values
class UniformValueSamplerGeneral:
    def __init__(self, space_dim, space_bounds, dev):
        self.space_dim = space_dim
        self.lower_bounds = space_bounds[:, 0]
        self.side_lengths = space_bounds[:, 1] - space_bounds[:, 0]

        assert len(self.lower_bounds) == space_dim
        assert len(self.side_lengths) == space_dim

        self.dev = dev

    def sample(self, batch_size):
        x = torch.rand([batch_size, self.space_dim], device = self.dev)
        values = self.lower_bounds + self.side_lengths * x
        return values
class NormalValueSampler:
    def __init__(self, space_dim, mean = 0, variance = 1):
        self.space_dim = space_dim
        self.mean = mean 
        self.variance = variance 

    def sample(self, batch_size):
        values = torch.tensor(self.variance).sqrt()* initial_values_sampler_gaussian(batch_size, self.space_dim) + self.mean
        return values
class BrownianPathSampler:
    # samples increments of brownian motion in space_dim-dimensional space.
    def __init__(self, space_dim, time_steps, dev):
        self.space_dim = space_dim
        self.time_steps = time_steps
        self.dev = dev

    def sample(self, batch_size):
        W = torch.randn([batch_size, self.space_dim, self.time_steps], device = self.dev)
        return W


class DataSampler:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.generator = iter(data_loader)

    def sample(self, batch_size):
        try:
            x = next(self.generator)
        except StopIteration:
            self.generator = iter(self.data_loader)
            x = next(self.generator)
        return x
class CubeSampler:
    def __init__(self, dim, dev):
        self.dim = dim
        self.dev = dev

    def sample(self, bs):
        x = initial_values_sampler_uniform(bs, self.dim, [-1., 1.])
        y = sample_from_boundary(self.dim, bs)
        return torch.cat([x, y], dim=1)
class BallSampler:
    def __init__(self, dim, dev):
        self.dim = dim
        self.dev = dev

    def sample(self, bs):
        x = sample_from_disk(self.dim, bs)
        y = sample_from_sphere(self.dim, bs)
        return torch.cat([x, y], dim=1)
    

class SkewSampler:
    """
        samples x from a normal distribution and then returns x * Softplus(x)
    """
    def __init__(self, space_dim, mean = 0, variance = 1):
        self.space_dim = space_dim
        self.mean = mean 
        self.variance = variance 

    def sample(self, batch_size):
        values = torch.tensor(self.variance).sqrt()* initial_values_sampler_gaussian(batch_size, self.space_dim) + self.mean 
        return values * nn.Softplus()(values)
    

class ExponentialSampler:
    """
        Samples from the exponential distribution lambda_ * exp(-lambda_ * x) where lambda_ > 0
    """
    def __init__(self,lambda_, device=dev):
        self.lambda_ = lambda_
        self.device = device

    def sample(self, batch_size):
        rate_tensor = torch.tensor(self.lambda_, device=self.device)
        samples = torch.distributions.Exponential(rate_tensor).sample((batch_size,))
        return samples 
    

class DrawBatch:
    # Draws a batch from given data tensor by randomly selecting indices.
    def __init__(self, data):
        self.data = data
        self.len = data.shape[0]

    def sample(self, bs):
        indices = torch.randint(0, self.len, (bs, ))
        return self.data[indices, :]
    




class BrownianMotionSampler:
    def __init__(self, d, ts,device = dev):
        self.d = d
        self.dt = ts - torch.roll(ts, 1)
        self.dt[0] = 0.
        self.N = ts.shape[0]
        self.device = device

    def sample(self, batch_size):
        w = torch.sqrt(self.dt) * torch.randn([batch_size, self.d, self.N],device = self.device)
        x = torch.cumsum(w, dim=-1)
        return x