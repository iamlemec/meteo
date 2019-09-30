# tensorflow tools

import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
T = tf.transpose

# funcs
def prod(a):
    return reduce(mul, [int(x) for x in a], 1)

def shape(a):
    return a.get_shape().as_list()

def size(a):
    return prod(shape(a))

def squeeze(a):
    return tf.reshape(a, [-1])

def flatify(a):
    return tf.concat([squeeze(x) for x in a], 0)

# def unpack(a, sh):
#     sz = [prod(s) for s in sh]
#     return [tf.reshape(x, s) for x, s in zip(tf.split(a, sz), sh)]

def unpack(a, sh):
    sz = np.cumsum([prod(s) for s in sh])
    return [np.reshape(x, s) for x, s in zip(np.split(a, sz), sh)]

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

def rename(x, name):
    return tf.identity(x, name=name)

def varname(nm):
    if ':' in nm: nm = ''.join(nm.split(':')[:-1])
    if '_' in nm: nm = ''.join(nm.split('_')[:-1])
    return nm

def summary(d):
    t = type(d)
    if t is dict:
        return {k.name: v for k, v in d.items()}

# y: function
# xs: list of variables
# out: [outputs, inputs]
def gradient(y, xs):
    with tf.GradientTape() as t:
        val = y()
    sz_x = [size(x) for x in xs]
    grd = t.gradient(val, xs, unconnected_gradients='zero')
    grd = [tf.reshape(g, [m]) for g, m in zip(grd, sz_x)]
    return tf.concat(grd, 0)

# y: model function
# xs: list of variables
# out: [outputs, inputs]
def jacobian(y, xs):
    with tf.GradientTape() as t:
        ys = y()
        vec = flatify(ys)
    sz_y = [size(z) for z in ys]
    sz_x = [size(z) for z in xs]
    n = sum(sz_y)
    jac = t.jacobian(vec, xs, unconnected_gradients='zero')
    jac = [tf.reshape(j, [n, m]) for j, m in zip(jac, sz_x)]
    return tf.concat(jac, 1)

def lstsq(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]
