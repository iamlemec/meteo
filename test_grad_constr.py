import sys
import tensorflow as tf
import numpy as np
from grad_constr import lagrange_objective, lagrange_objective2, constrained_gradient_descent, newton_solver, flatify, total_loss

# test
mod = sys.argv[1]

# algorithm params
maxiter = 20000
step = 0.2
lrate = 0.1
tol = 1.0e-7
per = 100

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
    obj = -tf.reduce_mean(x+y+z)
    con = [c1, c2]
    var = [x, y, z]

##
## growth Model
##

if mod == 'growth':
    # params
    rho = tf.Variable(0.05)
    lam = tf.Variable(0.2)
    c = tf.Variable(1.5)

    # eq vars
    wt = tf.Variable(0.9)
    vt = tf.Variable(1.1)
    tau = tf.Variable(0.12)
    P = tf.Variable(0.8)
    R = tf.Variable(0.2)

    # inter
    pit = lam/(1.0+lam)

    # cons
    lmc = P + R - 1.0
    val = (rho+tau)*vt - pit
    ent = wt*c - vt
    lab = (1.0+lam)*wt*P - 1.0
    res = R - tau*c

    # moments
    rnd = c*tau
    grw = tf.log(1.0+lam)*tau
    prf = pit
    mmt_gen = [rnd, grw, prf]
    mmt_dat = [0.18, 0.022, 0.17]
    mmt = [g - d for g, d in zip(mmt_gen, mmt_dat)]

    # output
    obj = total_loss(mmt)
    con = [lmc, val, ent, lab, res]
    var = [rho, lam, c, wt, vt, tau, P, R]

# update
# newt = newton_solver(con, var)

lobj, mult, lgrd_varz, lgrd_mult = lagrange_objective(obj, con, var)
cgd = tf.train.GradientDescentOptimizer(learning_rate=lrate)
mini = cgd.minimize(lobj, var_list=var+mult)

# lobj = lagrange_objective2(obj, con, 5.0)
# cgd = tf.train.GradientDescentOptimizer(learning_rate=lrate)
# mini = cgd.minimize(lobj, var_list=var)

# constraint error
cvec = flatify(con)
cerr = total_loss(con)

# output
def status(i):
    print(f'{i:4d}: {obj.eval():10g} {cerr.eval():10g}')

sess = tf.InteractiveSession()

print('initializing')
sess.run(tf.global_variables_initializer())
status(0)

# print('solving')
# for i in range(maxiter):
#     newt.run()
#     status(i)
#     if cerr.eval() < tol:
#         print('solved')
#         break

print('optimizing')
last_obj = np.inf
for i in range(maxiter):
    if i % per == 0:
        status(i)

    mini.run()

    next_obj = obj.eval()
    diff_obj = next_obj - last_obj
    last_obj = next_obj

    err = cerr.eval()

    if np.abs(next_obj) < tol and err < tol:
        print('optimized')
        status(i)
        break

    if np.isnan(next_obj) or np.isnan(err):
        print('failed')
        status(i)
        break
