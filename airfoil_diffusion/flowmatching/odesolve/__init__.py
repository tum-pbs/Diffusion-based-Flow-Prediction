import torch
from typing import Callable, Sequence, Union, Optional
from torch import Tensor
from dataclasses import dataclass
from enum import Enum
from .solver import *

class ODESolver(Enum):
    EULER = {"integrator": euler, "adaptive": False}
    MIDPOINT = {"integrator": midpoint, "adaptive": False}
    HEUN12 = {"integrator": heun12, "adaptive": True}
    RALSTON12 = {"integrator": ralston12, "adaptive": True}
    BOGACKI_SHAMPINE23 = {"integrator": bogacki_shampine23, "adaptive": True}
    RK4 = {"integrator": rk4, "adaptive": False}
    RK4_38RULE = {"integrator": rk4_38rule, "adaptive": False}
    DOPRI45 = {"integrator": dopri45, "adaptive": True}
    FEHLBERG45 = {"integrator": fehlberg45, "adaptive": True}
    CASHKARP45 = {"integrator": cashkarp45, "adaptive": True}

@dataclass
class FixedStepConfig:
    """
    Configuration for fixed-step ODE solver
    
    Args:
        solver (ODESolver): The ODE solver. Default to `ODESolver.DOPRI45`.
        dt (Tensor): The time step.
    """
    dt: Tensor
    solver: ODESolver = ODESolver.DOPRI45

class AdaptiveStepConfig:
    
    """
    Configuration for adaptive-step ODE solver
    
    Args:
        solver (ODESolver): The ODE solver. Default to `ODESolver.DOPRI45`.
        atol (float): The absolute tolerance for adaptive time stepping. Default to 1e-6.
        rtol (float): The relative tolerance for adaptive time stepping. Default to 1e-5.
    """
    
    def __init__(self, solver: ODESolver = ODESolver.DOPRI45,
                 atol: float = 1e-6, 
                 rtol: float = 1e-5) -> None:
        if solver.value["adaptive"] == False:
            adaptive_solvers = [solver.name for solver in ODESolver if solver.value["adaptive"]]
            raise ValueError(f"Solver '{solver.name}' does not support adaptive integration. Choose from {adaptive_solvers}")
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

def odeint(
    f: Callable[[Tensor, Tensor], Tensor],
    x_0: Tensor,
    t_0: Union[float, Tensor],
    t_1: Union[float, Tensor],
    solver_config: Union[FixedStepConfig, AdaptiveStepConfig],
    exchange_xt: bool = False,
    record_trajectory: bool = False,
    
) -> Union[Tensor, Sequence[Tensor]]:
    r"""
    Integrates a system of first-order ordinary differential equations (ODEs)

    $$
        \frac{dx}{dt} = f(t, x) ,
    $$
    
    The output is the final state

    $$
    x(t_1) = x_0 + \int_{t_0}^{t_1} f(t, x(t)) ~ dt .
    $$
    
    Modified from odeint function in zuko package

    Arguments:
        f (Callable[[Tensor, Tensor], Tensor]): A system of first-order ODEs :math:`f`.
        x (Tensor): The initial state :math:`x_0`.
        t_0 (Union[float, Tensor]): The initial integration time :math:`t_0`.
        t_1 (Union[float, Tensor]): The final integration time :math:`t_1`.
        solver_config (Union[FixedStepConfig, AdaptiveStepConfig]): The configuration for the ODE solver.
        exchange_xt: Whether to exchange the order of the arguments of the function `f`. Default to False.
            The input of the function `f` is `f(t, x)` if `exchange_xt` is False.
            The input of the function `f` is `f(x, t)` if `exchange_xt` is True
        record_trajectory (bool): Whether to record the trajectory. Default to False.
            If True, the function returns a list of states at each time step.
            If False, the function returns the final state.

    Returns:
        The final state :math:`x(t_1)`.

    Example:
        >>> A = torch.randn(3, 3)
        >>> f = lambda t, x: x @ A
        >>> x_0 = torch.randn(3)
        >>> x_1 = odeint(f, x_0, 0.0, 1.0)
        >>> x_1
        tensor([-1.4596,  0.5008,  1.5828])
    """
    adaptive= isinstance(solver_config, AdaptiveStepConfig)
    solver = solver_config.solver.value["integrator"]
    t_0 = torch.as_tensor(t_0, dtype=x_0.dtype, device=x_0.device)
    t_1 = torch.as_tensor(t_1, dtype=x_0.dtype, device=x_0.device)        
    if record_trajectory:
        trajectory = [x_0]
        
    with torch.no_grad():
        t=t_0
        x=x_0
        if adaptive:
            dt = t_1 - t_0
        else:
            dt = torch.as_tensor(solver_config.dt, dtype=x_0.dtype, device=x_0.device)
        sign = torch.sign(dt)
        while sign * (t_1 - t) > 0:
            dt = sign * torch.min(abs(dt), abs(t_1 - t))
            if adaptive:
                while True:
                    y, error = solver(f, x, t, dt, return_error=True)
                    tolerance = solver_config.atol + solver_config.rtol * torch.max(abs(x), abs(y))
                    error = torch.max(error / tolerance).clip(min=1e-9).item()
                    if error < 1.0:
                        x, t = y, t + dt
                        break
                    dt = dt * min(10.0, max(0.1, 0.9 / error ** (1 / 5)))
            else:
                x, t = solver(f, x, t, dt), t + dt    
            if record_trajectory:
                trajectory.append(x)
    if record_trajectory:
        return trajectory
    else:        
        return x

