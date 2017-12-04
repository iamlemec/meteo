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

def constrained_gradient_descent(obj, con, par, var, step=0.1):
    # shape info
    tot = par + var
    var_shp = [x.get_shape() for x in var]
    tot_shp = [x.get_shape() for x in tot]

    # derivatives
    con_vec = flatify(con)
    F = jacobian(obj, tot)
    G = jacobian(con_vec, tot)
    Gv = jacobian(con_vec, var)

    # constrained gradient descent
    H = tf.matmul(T(G), G)
    L = tf.matrix_solve(H, -tf.matmul(T(G), F))
    Ugd = step*tf.squeeze(F + tf.matmul(G, L))

    # correction step
    # J = tf.matmul(T(Gv), Gv)
    C = tf.expand_dims(con_vec, 1)
    # Ugn = tf.matrix_solve(J, -tf.matmul(T(Gv), C))
    Ugn = -step*tf.matmul(T(Gv), C)

    # updates
    gd_diffs = unpack(Ugd, tot_shp)
    gn_diffs = unpack(Ugn, var_shp)

    # operators
    gd_upds = increment(var, gd_diffs)
    gn_upds = increment(var, gn_diffs)

    return gd_upds, gn_upds

# init
x0 = 0.3
y0 = 0.3
z0 = np.sqrt(1.0-x0**2-y0**2)

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
cgd, gn = constrained_gradient_descent(f, [c1, c2], [z], [x, y], step=0.1)

cvec = tf.stack([c1, c2], 0)
cerr = tf.nn.l2_loss(cvec)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(x), sess.run(y), sess.run(z), sess.run(cerr))

    print('solving')
    for i in range(20):
        sess.run(gn)
        print(sess.run(x), sess.run(y), sess.run(z), sess.run(cerr))

    print('optimizing')
    for i in range(50):
        sess.run(cgd)
        sess.run(gn)
        print(sess.run(x), sess.run(y), sess.run(z), sess.run(cerr))
