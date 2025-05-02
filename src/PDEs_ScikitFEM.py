import time

import numpy as np
from scipy.sparse.linalg import splu, cg, bicg

try:
    from skfem import *
    from skfem.helpers import *
    from skfem.models.poisson import laplace, mass
except ImportError:
    print('skfem not installed.')



class FEM_Basis:
    def __init__(self, space_bounds, element_degree, spacediscr):
        self.spacedim = len(space_bounds)
        self.space_bounds = space_bounds
        self.nr_spacediscr = spacediscr + 1

        element_dict = {(1, 1): ElementLineP1(),
                        (1, 2): ElementLineP2(),
                        (2, 1): ElementTriP1(),
                        (2, 2): ElementTriP2(),
                        (3, 1): ElementTetP1(),
                        (3, 2): ElementTetP2()
                        }

        self.element = element_dict[(self.spacedim, element_degree)]
        self.element_name = self.element.__doc__
        self.element_degree = element_degree
        self.mesh, self.basis = create_rectangular_basis(space_bounds, self.element, spacediscr)
        self.sol_dim = len(self.basis.project(0.))

    def project_cont_function(self, function):
        u0 = self.basis.project(function)

        return u0

class DomainBasis:
    def __init__(self, domain='S', nr_refine=3, element_degree=1):
        if element_degree == 1:
            self.element = ElementTriP1()
        elif element_degree == 2:
            self.element = ElementTriP2()
        else:
            raise ValueError('element_degree must be 1 or 2.')

        if domain == 'S':
            self.mesh = MeshTri().init_tensor(np.linspace(-1., 1., 2 ** nr_refine + 1),
                                              np.linspace(-1., 1., 2 ** nr_refine + 1))
        elif domain == 'D':
            self.mesh = MeshTri().init_circle(nrefs=nr_refine)
        else:
            raise ValueError('domain must be S (square) or D (disk).')

        self.basis = Basis(self.mesh, self.element)


def compute_darcy_sol(base, conductivity, g):
    @BilinearForm
    def darcy_bilin(u, v, w):
        return w.k * dot(grad(u), grad(v))

    @LinearForm
    def rhs(v, _):
        return g * v

    b = asm(rhs, base.basis)
    A = asm(darcy_bilin, base.basis, k=base.basis.project(conductivity))

    A, b = enforce(A, b, D=base.basis.mesh.boundary_nodes())
    x = solve(A, b)
    return x


def allen_cahn_nonlin(inputs):
    return inputs - inputs * inputs * inputs


def sine_nonlin(inputs):
    return np.sin(inputs)


class ScikitFEM_PDE:
    def __init__(self, base, diffusivity, nonlin, nonlin_name, split_nonlin_flow):
        self.spacedim = base.spacedim
        self.sol_dim = base.sol_dim
        self.space_bounds = base.space_bounds

        self.element = base.element
        self.element_degree = base.element_degree
        self.element_name = base.element_name
        self.mesh = base.mesh
        self.basis = base.basis
        self.full_base = base

        self.nu = diffusivity
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        self.split_nonlin_flow = split_nonlin_flow

        L = - diffusivity * asm(laplace, self.basis)
        M = asm(mass, self.basis)
        self.Stiffness_operator, self.Mass_operator = penalize(L, M, D=self.basis.get_dofs())

    def compute_sol(self, reference_method, initial_values, T_end, nr_timesteps, params):
        u_final = reference_method(T_end, self, initial_values, nr_timesteps, params)

        return u_final


class ReferenceMethod:

    def __init__(self, reference_method, diffusivity, nonlin, nonlin_name, split_nonlin_flow, params=None, method_name="No name given"):
        self.reference_method = reference_method

        self.nu = diffusivity
        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        self.split_nonlin_flow = split_nonlin_flow
        self.params = params
        self.method_name = method_name

    def create_ode(self, base):
        ode = ScikitFEM_PDE(base, self.nu, self.nonlin, self.nonlin_name, self.split_nonlin_flow)
        return ode

    def compute_sol(self, T, initial_values, nr_timesteps, ode=None, base=None):
        if ode is None:
            ode = self.create_ode(base)

        reference_values = self.reference_method(T, ode, initial_values, nr_timesteps, self.params)

        return reference_values

    def get_reference_name(self):
        return self.method_name + " with params " + str(self.params)


def negative_third_power_flow(u, dt):
    denominator = np.sqrt(1. + 2. * dt * np.square(u))
    return u / denominator


def rational_approx_RK3_step(u, linear_op, mass, solver, dt):  # Crouzeix's diagonally implicit 3rd order RK method
    p1 = 0.5 + np.sqrt(3.) / 6.
    p2 = - np.sqrt(3.) / 3.

    b_one = linear_op @ u
    k_one = solver(b_one)

    b_two = linear_op @ ((1. - p1) * u + 0.5 * p2 * dt * k_one)
    v = solver(mass @ u + dt * b_two)

    return v


def Third_order_splitting_RK_FEM(T, fem_ode, initial_values, nr_timesteps, params):
    # solves PDE of the form u' = Lu + u + f(u) with splitting method, uses prescribed nonlinear flow to discretize f

    p1 = 0.5 + np.sqrt(3.) / 6.
    p2 = - np.sqrt(3.) / 3.
    alpha = params[0]
    timestep_size = float(T) / nr_timesteps

    L = fem_ode.Stiffness_operator
    M = fem_ode.Mass_operator

    linear_op = L + M

    time0 = time.perf_counter()
    solver = splu((M - timestep_size * p1 * linear_op).T).solve
    half_solver = splu((M - 0.5 * timestep_size * p1 * linear_op).T).solve
    time1 = time.perf_counter()

    u = initial_values
    for m in range(nr_timesteps):
        # Splitting scheme from [Jia & Li] (linear combination of Strang and symmetric splittings). This has order 3 iff alpha=2/3, otherwise order 2.
        u1 = fem_ode.split_nonlin_flow(u, 0.5 * timestep_size)
        u1 = rational_approx_RK3_step(u1, linear_op, M, solver, timestep_size)
        u1 = fem_ode.split_nonlin_flow(u1, 0.5 * timestep_size)

        u2 = rational_approx_RK3_step(u, linear_op, M, half_solver, 0.5 * timestep_size)
        u2 = fem_ode.split_nonlin_flow(u2, timestep_size)
        u2 = rational_approx_RK3_step(u2, linear_op, M, half_solver, 0.5 * timestep_size)

        u3 = rational_approx_RK3_step(u, linear_op, M, solver, timestep_size)
        u3 = fem_ode.split_nonlin_flow(u3, timestep_size)

        u4 = fem_ode.split_nonlin_flow(u, timestep_size)
        u4 = rational_approx_RK3_step(u4, linear_op, M, solver, timestep_size)

        u = alpha * (u1 + u2) + (0.5 - alpha) * (u3 + u4)

    time2 = time.perf_counter()

    times = [time1 - time0, time2 - time1]

    return u, times


def Second_order_linear_implicit_RK_FEM(T, fem_ode, initial_values, nr_timesteps, params):
    p1 = params[0]
    p2 = params[1]
    timestep_size = float(T) / nr_timesteps
    L = fem_ode.Stiffness_operator
    M = fem_ode.Mass_operator

    apply_solver = splu((M - timestep_size * p2 * L).T).solve

    u = initial_values
    for m in range(nr_timesteps):
        b_one = L @ u + M @ fem_ode.nonlin(u)
        k_one = apply_solver(b_one)

        b_two = L @ (u + 2 * p1 * (0.5 - p2) * timestep_size * k_one) + M @ fem_ode.nonlin(u + timestep_size * p1 * k_one)
        k_two = apply_solver(b_two)

        u = u + timestep_size * ((1 - 1. / (2 * p1)) * k_one + 1. / (2 * p1) * k_two)

    return u


def create_rectangular_basis(space_bounds, element, ncells):
    space_dim = len(space_bounds)

    if space_dim == 1:
        ref_mesh = MeshLine.init_tensor(np.linspace(0., space_bounds[0], 1 + ncells))

    elif space_dim == 2:
        ref_mesh = MeshTri.init_tensor(np.linspace(0., space_bounds[0], 1 + ncells), np.linspace(0., space_bounds[1], 1 + ncells))

    elif space_dim == 3:
        ref_mesh = MeshTet.init_tensor(np.linspace(0., space_bounds[0], 1 + ncells), np.linspace(0., space_bounds[1], 1 + ncells),
                                       np.linspace(0., space_bounds[2], 1 + ncells))

    else:
        raise ValueError('Space dimension must be 1, 2 or 3.')

    basis = Basis(ref_mesh, element)
    return ref_mesh, basis
