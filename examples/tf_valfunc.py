# tensorflow transition

from collections import Iterable
import tensorflow as tf
import numpy as np

# tuning params
K = 1000
sharp = 10.0
drag = 0.5
dt = 0.05

# constants
N = 100
kmin = 1.0
kmax = 10.0
kgrid = np.linspace(kmin, kmax, N+1).astype(np.float32)
kvec = 0.5*(kgrid[1:]+kgrid[:-1])
kwidth = np.diff(kgrid)

# utilities
def vderiv(v, meth='center'):
    if meth == 'left':
        d = (v[1:]-v[:-1])/(0.5*(kwidth[:-1]+kwidth[1:]))
        return tf.concat(0, [d[0:1], d])
    elif meth == 'right':
        d = (v[1:]-v[:-1])/(0.5*(kwidth[:-1]+kwidth[1:]))
        return tf.concat(0, [d, d[-1:]])

def sigmoid(v):
    return tf.sigmoid(sharp*v)

# clear any existing vars/ops
tf.reset_default_graph()

# parameters (as tf.Variable)
rho = tf.Variable(0.05, name='rho')
alpha = tf.Variable(0.3, name='alpha')
delta = tf.Variable(0.1, name='delta')

# state vector (as tf.Variable)
val = tf.Variable(3*np.linspace(0.0, 1.0, N), dtype=tf.float32, name='val')
inv = tf.Variable(np.zeros(N), dtype=tf.float32, name='inv')

# steady state
kss = (alpha/(rho+delta))**(1/(1-alpha))
iss = delta*kss
yss = kss**alpha
css = yss - iss
vkss = 1/css
vss = tf.log(css)/rho
sss = iss/yss

# initial value
y00 = tf.pow(kvec, alpha)
i00 = sss*y00
c00 = y00 - i00
u00 = tf.log(c00)
v00 = u00/rho

val_init = tf.assign(val, v00)
inv_init = tf.assign(inv, i00)
guess = tf.group(val_init, inv_init)

# derivatives - use upwinding scheme from moll et al
dk = inv - delta*kvec
vk_l = vderiv(val, meth='left')
vk_r = vderiv(val, meth='right')
vk = tf.sigmoid(-dk)*vk_l + tf.sigmoid(dk)*vk_r # this smoothness seems to promote stability

# choices
prod = kvec**alpha
cons = prod - inv
util = tf.log(cons)

# investment
inew = prod - 1 / vk

# value function
vnew = dt * util + ( 1 - dt * rho ) * ( val + dt * dk * vk )

# differentials and loss
val_diff = vnew - val
inv_diff = inew - inv
loss = tf.nn.l2_loss(val_diff) + tf.nn.l2_loss(inv_diff)

# optimizer
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# optim = opt.minimize(loss, var_list=[val, inv])

# operations
init = tf.global_variables_initializer()
v_upd = tf.assign(val, drag*val + (1-drag)*vnew)
i_upd = tf.assign(inv, drag*inv + (1-drag)*inew)
update = tf.group(v_upd, i_upd)

# store history
cstore = [
    'loss'
]
kstore = [
    'val', 'inv', 'prod', 'cons', 'util', 'dk',
    'vk', 'vk_l', 'vk_r',
    'val_diff', 'inv_diff'
]
store = cstore + kstore

scope = globals()
vardict = {n: scope[n] for n in store}
history = {}
history.update({n: np.zeros((K, 1)) for n in cstore})
history.update({n: np.zeros((K, N)) for n in kstore})

def store_hist(sess, i):
    for n in store:
        history[n][i,:] = np.squeeze(sess.run(vardict[n]))

rvec = np.arange(K)
def plot_hist(var, rep=None, val=None):
    if rep is not None:
        sr = rep
        sk = slice(None)
        x = kvec
    elif val is not None:
        sr = slice(None)
        sk = val
        x = rvec
    else:
        sr = slice(None)
        sk = 0
        x = rvec
    panes = len(var)
    (fig, axs) = plt.subplots(ncols=panes, figsize=(6*panes, 5))
    if not isinstance(axs, Iterable):
        axs = [axs]
    for (p, ax) in zip(var, axs):
        for n in p:
            ax.plot(x, history[n][sr,sk].T)
        ax.legend(p)
        sns.despine(ax=ax)

# run loops
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(kss))
    for i in range(K):
        sess.run(update)
        store_hist(sess, i)

