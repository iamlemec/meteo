import tensorflow as tf
import numpy as np
from tools import *

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
    L = tf.linalg.solve(
        tf.matmul(T(G), G),
       -tf.matmul(T(G), F)
    )
    Ugd = step*squeeze(F + tf.matmul(G, L))

    # correction step (zangwill-garcia)
    # can be non-square so use least squares
    Ugz = squeeze(tf.linalg.lstsq(
        tf.concat([T(G), Ugd[None, :]], 0),
       -tf.concat([g[:, None], [[0.0]]], 0),
        fast=False
    ))

    # updates
    gd_diffs = unpack(Ugd, var_shp)
    gz_diffs = unpack(Ugz, var_shp)

    # operators
    gd_upds = increment(var, gd_diffs)
    gz_upds = increment(var, gz_diffs)

    # slope
    gain = tf.squeeze(tf.matmul(Ugd[None, :], F))

    return gd_upds, gz_upds, gain

def newton_solver(con, var):
    # shape info
    var_shp = [x.get_shape() for x in var]

    # derivatives
    g = flatify(con)
    G = jacobian(g, var)

    # can be non-square so use least squares
    U = squeeze(tf.matrix_solve_ls(T(G), -g[:, None], fast=False))

    # updates
    diffs = unpack(U, var_shp)

    # operators
    upds = increment(var, diffs)

    return upds
