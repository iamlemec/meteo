import tensorflow as tf
import numpy as np
import numpy.linalg as la
from operator import mul
from functools import reduce
T = tf.transpose

# params
step = 0.1

# funcs
def tolist(a):
    t = type(a)
    if t is tuple:
        return list(a)
    elif t is list:
        return a
    else:
        return [a]

def prod(a):
    return reduce(mul, a)

# dim 0: inputs
# dim 1: outputs
def gradv(a, b):
    a = tolist(a)
    b = tolist(b)
    return tf.stack([tf.stack([tf.gradients(w, z)[0] for z in b], axis=0) for w in a], axis=1)

class ConstrainedGradientDescent:
    def __init__(self, obj, var):
        self.obj = obj
        self.var = var

        self.siz = [x.get_shape() for x in self.var]
        self.len = [prod(x) for x in self.siz]


# init
x0 = 0.2
y0 = np.sqrt(1.0-x0**2)

# vars
x = tf.Variable(x0, dtype=tf.float64)
y = tf.Variable(y0, dtype=tf.float64)
v = [x, y]
N = len(v)

# cons
c = 1.0 - x**2 - y**2
g = [c]
M = len(g)

# obj
f = x + y

# grads
F = gradv(f, v)
G = gradv(g, v)

# step
I = tf.eye(N, dtype=tf.float64)
L = tf.matrix_solve(tf.matmul(T(G), G), -tf.matmul(T(G), F))
U = tf.squeeze(F + tf.matmul(G, L))

# update
Ux = step*U[0]
Uy = step*U[1]
Rx = x.assign_add(Ux)
Ry = y.assign_add(Uy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(x), sess.run(y))

    for i in range(50):
        sess.run([Rx, Ry])
        print(i, sess.run(x), sess.run(y))
