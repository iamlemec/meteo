import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
T = tf.transpose

maxiter = 100

# funcs
def prod(a):
    return reduce(mul, [int(x) for x in a], 1)

def size(a):
    return prod(a.get_shape())

def flatify(a):
    return tf.concat([tf.reshape(x, [-1]) for x in a], 0)

def unpack(a, sh):
    sz = [prod(s) for s in sh]
    return [tf.reshape(x, s) for x, s in zip(tf.split(a, sz), sh)]

def increment(a, b):
    return tf.group(*[x.assign_add(u) for x, u in zip(a, b)])

# dim 0: inputs
# dim 1: outputs
def jacobian(a, b):
    n = size(a)
    if n > 1:
        return tf.stack([flatify(tf.gradients(a[i], b)) for i in range(n)], axis=1)
    else:
        return tf.stack([flatify(tf.gradients(a, b))], axis=1)

# N: number of vars
# M: number of cons
# G: N x M
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
    Ugd = step*tf.squeeze(F + tf.matmul(G, L))

    # correction step (zangwill-garcia)
    Ugz = tf.squeeze(tf.matrix_solve(
        tf.concat([T(G), Ugd[None, :]], 0),
       -tf.concat([g[:, None], [[0.0]]], 0)
    ))

    # updates
    gd_diffs = unpack(Ugd, var_shp)
    gz_diffs = unpack(Ugz, var_shp)

    # operators
    gd_upds = increment(var, gd_diffs)
    gz_upds = increment(var, gz_diffs)

    return gd_upds, gz_upds

# init
x0 = 0.3
y0 = 0.3
z0 = 0.3

# vars
x = tf.Variable(x0, dtype=tf.float64)
y = tf.Variable(y0, dtype=tf.float64)
z = tf.Variable(z0, dtype=tf.float64)

# cons
c1 = 1.0 - (x-0.5)**2 - (y+0.5)**2 - z**2
c2 = 1.0 - (x+0.5)**2 - (y-0.5)**2 - z**2

# obj
f = x + y + z

# update
cgd, gn = constrained_gradient_descent(f, [c1, c2], [x, y, z], step=0.1)

# constraint error
cvec = flatify([c1, c2])
cerr = tf.sqrt(tf.reduce_mean(cvec**2))

with tf.Session() as sess:
    print('initializing')
    sess.run(tf.global_variables_initializer())
    print(f'{i:3d}: {x.eval():10g} {y.eval():10g} {z.eval():10g} = {f.eval():10g} {cerr.eval():10g}')

    print('solving')
    for i in range(maxiter):
        sess.run(gn)
        err0 = cerr.eval()
        print(f'{i:3d}: {x.eval():10g} {y.eval():10g} {z.eval():10g} = {f.eval():10g} {err0:10g}')
        if err0 < 1e-14:
            break

    print('optimizing')
    for i in range(maxiter):
        sess.run(cgd)
        sess.run(gn)
        err0 = cerr.eval()
        print(f'{i:3d}: {x.eval():10g} {y.eval():10g} {z.eval():10g} = {f.eval():10g} {err0:10g}')
        if err0 < 1e-14:
            break
