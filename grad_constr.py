import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
T = tf.transpose

# funcs
def prod(a):
    return reduce(mul, [int(x) for x in a], 1)

def size(a):
    return prod(a.get_shape())

def squeeze(a):
    return tf.reshape(a, [-1])

def flatify(a):
    return tf.concat([squeeze(x) for x in a], 0)

def unpack(a, sh):
    sz = [prod(s) for s in sh]
    return [tf.reshape(x, s) for x, s in zip(tf.split(a, sz), sh)]

def increment(a, b):
    return tf.group(*[x.assign_add(u) for x, u in zip(a, b)])

def grad(a, b):
    for x in b:
        g = tf.gradients(a, x)[0]
        yield g if g is not None else tf.zeros(x.get_shape())

# dim 0: inputs
# dim 1: outputs
def jacobian(a, b):
    n = size(a)
    if n > 1:
        return tf.stack([flatify(grad(a[i], b)) for i in range(n)], axis=1)
    else:
        return tf.stack([flatify(grad(a, b))], axis=1)

# N: number of vars
# M: number of cons
# G: N x M
# F: N x 1
def constrained_gradient_descent(obj, con, var, step=0.1):
    # shape info
    var_shp = [x.get_shape() for x in var]

    # derivatives
    g = flatify(con)
    F = jacobian(obj, var)
    G = jacobian(g, var)

    # constrained gradient descent
    L = tf.matrix_solve(
        tf.matmul(T(G), G),
       -tf.matmul(T(G), F)
    )
    Ugd = step*squeeze(F + tf.matmul(G, L))

    # correction step (zangwill-garcia)
    # can be non-square so use least squares
    Ugz = squeeze(tf.matrix_solve_ls(
        tf.concat([T(G), Ugd[None, :]], 0),
       -tf.concat([g[:, None], [[0.0]]], 0)
    ))

    # updates
    gd_diffs = unpack(Ugd, var_shp)
    gz_diffs = unpack(Ugz, var_shp)

    # operators
    gd_upds = increment(var, gd_diffs)
    gz_upds = increment(var, gz_diffs)

    return gd_upds, gz_upds, F

def newton_solver(con, var):
    # shape info
    var_shp = [x.get_shape() for x in var]

    # derivatives
    g = flatify(con)
    G = jacobian(g, var)

    # can be non-square so use least squares
    U = squeeze(tf.matrix_solve_ls(T(G), -g[:, None]))

    # updates
    diffs = unpack(U, var_shp)

    # operators
    upds = increment(var, diffs)

    return upds
