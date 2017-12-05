import sys
import tensorflow as tf
import numpy as np
from grad_constr import constrained_gradient_descent, newton_solver, flatify

# test
mod = sys.argv[1]

# algorithm params
maxiter = 500
step = 0.2
tol = 1.0e-7

##
## algebraic system
##

if mod == 'algebraic':
    # init
    x0 = 0.1
    y0 = -0.2
    z0 = 0.3

    # vars
    x = tf.Variable(3*[x0])
    y = tf.Variable(3*[y0])
    z = tf.Variable(3*[z0])

    # cons
    c1 = 1.0 - (x-0.5)**2 - (y+0.5)**2 - z**2
    c2 = 1.0 - (x+0.5)**2 - (y-0.5)**2 - z**2

    # output
    obj = tf.reduce_mean(x+y+z)
    con = [c1, c2]
    var = [x, y, z]

##
## growth Model
##

if mod == 'growth':
    # params
    rho = tf.Variable(0.05)
    ilam = tf.Variable(0.9)
    c = tf.Variable(1.5)
    eta = tf.Variable(2.2)
    F = tf.Variable(0.1)

    # eq vars
    wt = tf.Variable(0.9)
    vt = tf.Variable(1.1)
    x = tf.Variable(0.1)
    e = tf.Variable(0.3)
    tau = tf.Variable(0.15)
    P = tf.Variable(0.8)
    R = tf.Variable(0.2)

    # inter
    C = c*x**eta
    Cp = c*eta*x**(eta-1.0)
    pit = 1.0 - ilam

    # cons
    lmc = P + R - 1.0
    val = (rho+tau)*vt - pit
    foc = wt*Cp - vt
    ent = wt*F - (x*vt-wt*C)
    des = tau - (1.0+e)*x
    lab = wt*P - ilam
    res = (1.0+e)*C + e*F

    # moments
    rnd = wt*C
    grw = -tf.log(ilam)*tau
    prf = pit - rnd
    ent = e/(1.0+e)
    mmt = [rnd-0.1, grw-0.03, prf-0.15, ent-0.3]
    mvec = flatify(mmt)

    # output
    obj = -tf.reduce_sum(mvec**2)
    con = [lmc, val, foc, ent, des, lab, res]
    var = [rho, ilam, c, eta, F, wt, vt, x, e, tau, P, R]

# update
newt = newton_solver(con, var)
pred, corr, gain = constrained_gradient_descent(obj, con, var, step=step)

# constraint error
cvec = flatify(con)
cerr = tf.sqrt(tf.reduce_mean(cvec**2))

# output
def status(i):
    print(f'{i:4d}: {obj.eval():10g} {cerr.eval():10g} {gain.eval():10g}')

# with tf.Session() as sess:

sess = tf.InteractiveSession()

print('initializing')
sess.run(tf.global_variables_initializer())
status(0)

print('solving')
for i in range(maxiter):
    newt.run()
    status(i)
    if cerr.eval() < tol:
        break

print('optimizing')
for i in range(maxiter):
    pred.run()
    corr.run()
    if i % 10 == 0: status(i)
    if gain.eval() < tol:
        status(i)
        break
