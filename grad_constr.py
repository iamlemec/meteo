import tensorflow as tf
import numpy as np
from tools import *

# M: number of cons
# N: number of vars
# G: M x N
# F: N x 1
def constrained_gradient_descent(obj, con, var, step=0.1, tol=1e-5, max_iter=100, corr_steps=1, output=False):
    # shape info
    var_shp = [shape(x) for x in var]

    for i in range(max_iter):
        # derivatives
        g = flatify(con()).numpy()
        G = jacobian(con, var).numpy()
        F = gradient(obj, var).numpy()

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
        g = flatify(con()).numpy()
        G = jacobian(con, var).numpy()

        # correction step (zangwill-garcia), can be non-square so use least squares
        corr = lstsq(
            np.vstack([G, grad[None, :]]),
           -np.r_[g, 0.0]
        )

        # increment
        corr_diffs = unpack(corr, var_shp)
        increment(var, corr_diffs)

        # derivatives
        g = flatify(con()).numpy()

        # error
        move = np.max(np.abs(gain))
        err = np.max(np.abs(g))

        # output
        if output and i % 10 == 0:
            print(f'{i:4d}: move = {move:7.5f}, err = {err:7.5f}')

        # convergence
        if move < tol and err < tol:
            if output:
                print(f' FIN: move = {move:7.5f}, err = {err:7.5f}')
            break
