# tensorflow tools

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

def assign(a, b):
    return tf.group(*[x.assign(u) for x, u in zip(a, b)])

def increment(a, b, d=1):
    return tf.group(*[x.assign_add(d*u) for x, u in zip(a, b)])

def grad(a, b):
    for x in b:
        g = tf.gradients(a, x)[0]
        yield g if g is not None else tf.zeros(x.get_shape())

def total_loss(vec):
    return tf.reduce_sum([tf.nn.l2_loss(v) for v in vec])

# a: tensor
# b: list of variables
# out: [inputs, outputs]
def jacobian(a, b):
    n = size(a)
    if n > 1:
        return tf.stack([flatify(grad(a[i], b)) for i in range(n)], 1)
    else:
        return tf.stack([flatify(grad(a, b))], 1)

# one step in newton's method
# eqn: tensor
# var: list of variables
def newton_step(eqn, var, jac=None):
    # shape info
    var_shp = [x.get_shape() for x in var]

    # derivatives
    if jac is None:
        jac = jacobian(eqn, var)

    # can be non-square so use least squares
    step = squeeze(tf.linalg.lstsq(T(jac), -eqn[:, None], fast=False))

    # updates
    diffs = unpack(step, var_shp)

    # operators
    upds = increment(var, diffs)

    return upds
