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

def increment(a, b):
    return tf.group(*[x.assign_add(u) for x, u in zip(a, b)])

def grad(a, b):
    for x in b:
        g = tf.gradients(a, x)[0]
        yield g if g is not None else tf.zeros(x.get_shape())

def total_loss(vec):
    return tf.reduce_sum([tf.nn.l2_loss(v) for v in vec])

# dim 0: inputs
# dim 1: outputs
def jacobian(a, b):
    n = size(a)
    if n > 1:
        return tf.stack([flatify(grad(a[i], b)) for i in range(n)], axis=1)
    else:
        return tf.stack([flatify(grad(a, b))], axis=1)
