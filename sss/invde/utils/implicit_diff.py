### Implicit differentiation
# when doing inverse design with a PDE governed system A(x)y = b, 
# in which y is solution (E field), x is design variables (dielectric distribution)
# when we have a loss function l(y), we want to compute gradient dl/dx
# However, if we use iterative methods to solve A(x)y = b, usually we can't use auto-diff directly,
# because the computational graph is too large.

# instead, we can use implicit differentiation to compute the gradient.
# define F(y(x), x) = A(x)y(x) - b = 0

# let's say x is a vector of length n, y is a vector of length m, A has length m*m, F(y, x) has length m
# first,    dF/dx = ∂F/∂y * dy/dx + ∂F/∂x = 0 => dy/dx = -(∂F/∂y)^-1 * ∂F/∂x
# shape:    (m,n) = (m,m) * (m,n) + (m,n)        (m,n) = (m,m) * (m,n)

# so,       dl/dx =  - dl/dy * (∂F/∂y)^-1 * ∂F/∂x
# shape:    (1,n) =    (1,m) *  (m,m)     * (m,n)

# define v, adjoint variable of y:
#           v = (∂F/∂y)^-T * (dl/dy)^T
# shape:(m,1) =  (m,m)     *  (m,1)
# equivalently:
#         A v = (dl/dy)^T

# then,     dl/dx = - v^T * dF/dx
# shape:    (1,n) = (1,m) * (m,n)

import torch
import gin

@gin.configurable
def implicit_diff(
    rhs,
    design_param,
    full_eps_fn,
    iterative_solver,
    residual_fn,
    loss_fn,
    **kwargs
):
    """
    rhs: right hand side of the equation A(x)y = b
    design_param: design parameters (requires grad)
    full_eps_fn: function to compute the full eps from design_param
    iterative_solver: iterative solver to solve y = iterative_solver(rhs, eps)
    residual_fn: function to compute the residual: residual = F(y, x) = A(x)y(x) - rhs
    loss_fn: loss function
    """
    with torch.no_grad(): # no gradient for forward pass
        eps = full_eps_fn(design_param)
        y = iterative_solver(rhs, eps, **kwargs)

    loss = loss_fn(y)
    dl_dy = torch.autograd.grad(loss, y)[0]

    # solve the adjoint equation
    with torch.no_grad():
        y_adj = iterative_solver(dl_dy, eps, **kwargs)

    F = residual_fn(y, design_param) # F(y, x) = A(x)y(x) - rhs
    dF_dx = torch.autograd.grad(F, design_param)[0]

    dl_dx = -y_adj.T @ dF_dx

    return dl_dx
    
