from jax import jit, grad, jacobian
import jax.numpy as np
np.concat = np.concatenate
from operator import mul
from functools import reduce
import numpy as np0
import inspect

# reshaping

def flatify(a):
    return np.concat([x.flatten() for x in a])

def boxify(a):
    return np.vstack([np.hstack(z) for z in a])

def arrayify(x):
    return np.array(x, ndmin=1)

def scalarify(x):
    if x.size == 1:
        return np0.asscalar(x)
    else:
        return np0.array(x)

# jax wrappers

def get_args(f):
    return inspect.getfullargspec(f)[0]

def get_nargs(f):
    return len(get_args(f))

def fastvec(f):
    v = lambda *x: flatify(f(*x))
    return jit(v)

def fastjac(f, argnums=None):
    if argnums is None:
        narg = get_nargs(f)
        argnums = range(narg)
    j0 = jacobian(f, argnums=argnums)
    j = lambda *x: boxify(j0(*x))
    return jit(j)

def fastgrad(f, argnums=None):
    if argnums is None:
        narg = get_nargs(f)
        argnums = range(narg)
    g0 = grad(f, argnums=argnums)
    g = lambda *x: np.concat(g0(*x))
    return jit(g)

# variable managment

def prod(a):
    return reduce(mul, [int(x) for x in a], 1)

def unpack(v, sh):
    sz = np0.cumsum([prod(s) for s in sh])
    return [np.reshape(x, s) for x, s in zip(np.split(v, sz), sh)]

def increment(v, x):
    for i in range(len(v)):
        v[i] += x[i]

def bound(v, vmin, vmax):
    for i in range(len(v)):
        if vmin[i] is not None:
            v[i] += np.maximum(0.0, vmin[i]-v[i])
        if vmax[i] is not None:
            v[i] += np.minimum(0.0, vmax[i]-v[i])

# linear algebra

def lstsq(A, b):
    return np0.linalg.lstsq(A, b, rcond=None)[0]

##
## gradient descent (unconstrained)
##

# N: number of vars
# F: N x 1
def gradient_descent(obj, var, vmin={}, vmax={}, vnames=None, step=0.1, tol=1e-5, max_iter=1000, output=False):
    # map dictionary
    if vnames is None:
        vnames = get_args(obj)
    argnums = range(len(vnames))
    var = [arrayify(var[n]) for n in vnames]
    vshp = [x.shape for x in var]

    # bounds
    vmin = [vmin.get(n, None) for n in vnames]
    vmax = [vmax.get(n, None) for n in vnames]

    # jax magic
    f_fun = jit(obj)
    F_fun = fastgrad(obj, argnums=argnums)

    for rep in range(max_iter):
        # derivatives
        F = F_fun(*var)

        if np.isnan(F).any():
            print(f'{rep:4d}: encountered invalid values')
            break

        grad = -step*F
        gain = np.dot(grad, F)

        # increment with bounds
        diffs = unpack(grad, vshp)
        increment(var, diffs)
        bound(var, vmin, vmax)

        # values
        f = f_fun(*var)

        # error
        move = np.max(np.abs(gain))

        # output
        if output and rep % 100 == 0:
            print(f'{rep:4d}: val = {f:7.5f}')

        # convergence
        if move < tol:
            if output:
                print(f'{rep:4d}: val = {f:7.5f}')
            break

    # return solution
    var1 = [scalarify(x) for x in var]
    return {n: v for n, v in zip(vnames, var1)}

##
## least squares solver
##

l2_norm = lambda x: np.sum(x*x)

def gradient_lstsq(eqn, var, **kwargs):
    vnames = get_args(eqn)
    obj2 = lambda *x: l2_norm(flatify(eqn(*x)))
    return gradient_descent(obj2, var, vnames=vnames, **kwargs)

##
## constrained gradient descent
##

# M: number of cons
# N: number of vars
# G: M x N
# F: N x 1
def constrained_gradient_descent(obj, con, var, vmin={}, vmax={}, step=0.1, tol=1e-5, max_iter=1000, output=False):
    # these need to align
    vnames = get_args(obj)
    vnames_con = get_args(con)
    assert(vnames == vnames_con)

    # map dictionary
    var = [arrayify(var[n]) for n in vnames]
    vmin = [vmin.get(n, None) for n in vnames]
    vmax = [vmax.get(n, None) for n in vnames]

    # variable info
    var_shp = [x.shape for x in var]

    # jax magic
    g_fun = fastvec(con)
    f_fun = jit(obj)
    G_fun = fastjac(con)
    F_fun = fastgrad(obj)

    for rep in range(max_iter):
        # derivatives
        g = g_fun(*var)
        G = G_fun(*var)
        F = F_fun(*var)

        if np.isnan(g).any() or np.isnan(G).any() or np.isnan(F).any():
            print(f'{rep:4d}: encountered invalid values')
            break

        # constrained gradient descent
        L = np.linalg.solve(
            np.matmul(G, G.T),
           -np.dot(G, F)
        )
        grad = -step*(F + np.dot(G.T, L))
        gain = np.dot(grad, F)

        # increment with bounds
        grad_diffs = unpack(grad, var_shp)
        increment(var, grad_diffs)
        bound(var, vmin, vmax)

        # derivatives
        g = g_fun(*var)
        G = G_fun(*var)

        if np.isnan(g).any() or np.isnan(G).any():
            print(f'{rep:4d}: encountered invalid values')
            break

        # correction step (zangwill-garcia), can be non-square so use least squares
        corr = lstsq(
            np.vstack([G, grad[None, :]]),
           -np.concat([g, np.array([0.0])])
        )

        # increment with bounds
        corr_diffs = unpack(corr, var_shp)
        increment(var, corr_diffs)
        bound(var, vmin, vmax)

        # derivatives
        f = f_fun(*var)
        g = g_fun(*var)

        # error
        move = np.max(np.abs(gain))
        err = np.max(np.abs(g))

        # output
        if output and rep % 100 == 0:
            print(f'{rep:4d}: val = {f:7.5f}, err = {err:7.5f}')

        # convergence
        if move < tol and err < tol:
            if output:
                print(f'{rep:4d}: val = {f:7.5f}, err = {err:7.5f}')
            break

    # return solution
    var1 = [scalarify(x) for x in var]
    return {n: v for n, v in zip(vnames, var1)}
