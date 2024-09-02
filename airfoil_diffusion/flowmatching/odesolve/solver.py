from typing import Callable, Optional,Sequence
from torch import Tensor

def rk_step(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    ca: Sequence,
    b: Sequence,
    b_star: Optional[Sequence]=None,
    return_error: bool = False
    )->Tensor:
    """
    Runge-Kutta step with a general Butcher tableau.
    The number of function evaluations is len(ca)+1.
    
    Args:
        f (Callable[[Tensor, Tensor], Tensor]): function to integrate
        x: (Tensor) current state
        t: (Tensor) current time
        dt (Tensor): time step
        ca (Tensor): coefficients for the Runge-Kutta method
        b (Tensor): coefficients for the Runge-Kutta method
        b_star (Tensor): coefficients for the error estimate
        return_error (bool): whether to return the error estimate
    
    Returns:
        x_new: new state
    
    """
    ks=[f(t,x)]
    for ca_i in ca:
        ks.append(f(t+ca_i[0]*dt,
                    x+dt*sum([a_i*k for a_i,k in zip(ca_i[1:],ks)])))
    x_new=x+dt*sum([b_i*k for b_i,k in zip(b,ks)])
    if b_star is not None and return_error:
        b_dif=[b_i-b_star_i for b_i,b_star_i in zip(b,b_star)]
        error=dt*sum([b_dif_i*k for b_dif_i,k in zip(b_dif,ks)])
        return x_new,error
    return x_new

def euler(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
):
    """ First order Euler method. NFE=1"""
    return rk_step(f,x,t,dt,[],[1])

def midpoint(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
):
    """ Second order midpoint method. NFE=2 """
    return rk_step(f,x,t,dt,[[1/2,1/2]],[0,1])

def heun12(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    return_error: bool = False,
):
    """ Second order Heun's method. NFE=2 """
    return rk_step(f,x,t,dt,[[1,1]],[1/2,1/2],[1,0],return_error)

def ralston12(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    return_error: bool = False,
):
    """ Second order Ralston's method. NFE=2 """
    return rk_step(f,x,t,dt,[[2/3,2/3]],[1/4,3/4],[2/3,1/3],return_error)

def bogacki_shampine23(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    return_error: bool = False,
):
    """ Third order Bogacki-Shampine method. NFE=4 """
    ca=[
        [1/2,1/2],
        [3/4,0,3/4],
        [1,2/9,1/3,4/9],
    ]
    b=[2/9,1/3,4/9,0]
    b_star=[7/24,1/4,1/3,1/8]
    return rk_step(f,x,t,dt,ca,b,b_star,return_error)

def rk4(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
):
    """ Fourth order Runge-Kutta method. NFE=4 """
    ca=[
        [1/2,1/2],
        [1/2,0,1/2],
        [1,0,0,1],
    ]
    b=[1/6,1/3,1/3,1/6]
    return rk_step(f,x,t,dt,ca,b)

def rk4_38rule(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
):
    """ Fourth order Runge-Kutta method with 3/8 rule. NFE=4 """
    ca=[
        [1/3,1/3],
        [2/3,-1/3,1],
        [1,-1,1,1],
    ]
    b=[1/8,3/8,3/8,1/8]
    return rk_step(f,x,t,dt,ca,b)

def dopri45(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    return_error: bool = False,
):
    """ Fifth order Dormand-Prince method. NFE=7 """
    ca=[
        [1/5,1/5],
        [3/10,3/40,9/40],
        [4/5,44/45,-56/15,32/9],
        [8/9,19372/6561,-25360/2187,64448/6561,-212/729],
        [1,9017/3168,-355/33,46732/5247,49/176,-5103/18656],
        [1,35/384,0,500/1113,125/192,-2187/6784,11/84]
    ]
    b=[35/384,0,500/1113,125/192,-2187/6784,11/84]
    b_star=[5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,1/40]
    return rk_step(f,x,t,dt,ca,b,b_star,return_error)

def fehlberg45(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    return_error: bool = False,
):
    """ Fifth order Fehlberg method. NFE=6 """
    ca=[
        [1/4,1/4],
        [3/8,3/32,9/32],
        [12/13,1932/2197,-7200/2197,7296/2197],
        [1,439/216,-8,3680/513,-845/4104],
        [1/2,-8/27,2,-3544/2565,1859/4104,-11/40]
    ]
    b=[16/135,0,6656/12825,28561/56430,-9/50,2/55]
    b_star=[25/216,0,1408/2565,2197/4104,-1/5,0]
    return rk_step(f,x,t,dt,ca,b,b_star,return_error)

def cashkarp45(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    t: Tensor,
    dt: Tensor,
    return_error: bool = False,
):
    """ Fifth order Cash-Karp method. NFE=6 """
    ca=[
        [1/5,1/5],
        [3/10,3/40,9/40],
        [3/5,3/10,-9/10,6/5],
        [1,-11/54,5/2,-70/27,35/27],
        [7/8,1631/55296,175/512,575/13824,44275/110592,253/4096]
    ]
    b=[37/378,0,250/621,125/594,0,512/1771]
    b_star=[2825/27648,0,18575/48384,13525/55296,277/14336,1/4]
    return rk_step(f,x,t,dt,ca,b,b_star,return_error)