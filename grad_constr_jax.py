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

def unpack(a, sh):
    sz = np0.cumsum([prod(s) for s in sh])
    return [np.reshape(x, s) for x, s in zip(np.split(a, sz), sh)]

def increment(a, b):
    for i in range(len(a)):
        a[i] += b[i]

# linear algebra

def lstsq(A, b):
    return np0.linalg.lstsq(A, b, rcond=None)[0]

# M: number of cons
# N: number of vars
# G: M x N
# F: N x 1
def constrained_gradient_descent(obj, con, var, step=0.1, tol=1e-5, max_iter=100, corr_steps=1, output=False):
    # these need to align
    vnames = get_args(obj)
    vnames_con = get_args(con)
    assert(vnames == vnames_con)

    # map dictionary
    var = [var[n] for n in vnames]

    # variable info
    var_shp = [x.shape for x in var]

    # jax magic
    g_fun = fastvec(con)
    G_fun = fastjac(con)
    F_fun = fastgrad(obj)

    for rep in range(max_iter):
        # derivatives
        g = g_fun(*var)
        G = G_fun(*var)
        F = F_fun(*var)

        # constrained gradient descent
        L = np.linalg.solve(
            np.matmul(G, G.T),
           -np.dot(G, F)
        )
        grad = step*(F + np.dot(G.T, L))
        gain = np.dot(grad, F)

        # increment
        grad_diffs = unpack(grad, var_shp)
        increment(var, grad_diffs)

        # derivatives
        g = g_fun(*var)
        G = G_fun(*var)

        # correction step (zangwill-garcia), can be non-square so use least squares
        corr = lstsq(
            np.vstack([G, grad[None, :]]),
           -np.concat([g, np.array([0.0])])
        )

        # increment
        corr_diffs = unpack(corr, var_shp)
        increment(var, corr_diffs)

        # derivatives
        g = g_fun(*var)

        # error
        move = np.max(np.abs(gain))
        err = np.max(np.abs(g))

        # output
        if output and rep % 10 == 0:
            print(f'{rep:4d}: move = {move:7.5f}, err = {err:7.5f}')

        # convergence
        if move < tol and err < tol:
            if output:
                print(f'{rep:4d}: move = {move:7.5f}, err = {err:7.5f}')
            break

    # return solution
    return {n: v for n, v in zip(vnames, var)}
