# tensorflow transition

from itertools import chain
import tensorflow as tf
import numpy as np

# tuning params
K = 5000
sharp = 10.0
cmin = 0.01
drag = 0.9
dt = 0.05

# capital grid
N = 101
nmid = N//2
amin = -10.0
amax = 20.0
agrid = np.linspace(amin, amax, N+1).astype(np.float32)
avec = 0.5*(agrid[1:]+agrid[:-1])
awidth = np.diff(agrid)

# state grid
M = 2

# utilities
def vderiv(v, meth='center'):
    if meth == 'left':
        d = (v[:,1:]-v[:,:-1])/(0.5*(awidth[:-1]+awidth[1:]))
        return tf.concat(1, [d[:,0:1], d])
    elif meth == 'right':
        d = (v[:,1:]-v[:,:-1])/(0.5*(awidth[:-1]+awidth[1:]))
        return tf.concat(1, [d, d[:,-1:]])

def sigmoid(v, s=sharp):
    return tf.sigmoid(s*v)

def smoothmax(v1, v2, sharp=sharp):
    vdiff = v1 - v2
    return sigmoid(vdiff, sharp)*v1 + sigmoid(-vdiff, sharp)*v2

def maxplus(v, offset, sharp=sharp):
    return (tf.nn.softplus(sharp*v-offset)+offset)/sharp

# clear any existing vars/ops
tf.reset_default_graph()

# parameters (as tf.Variable)
rho = tf.Variable(0.05, name='rho')
irate = tf.Variable(np.kron(np.ones((M, 1)), np.r_[np.linspace(0.1, 0.05, N//2), np.linspace(0.05, 0.05, N//2+1)]), dtype=tf.float32, name='irate')
wage = tf.Variable(np.linspace(0.5, 1.0, M).reshape((M, 1)), dtype=tf.float32, name='wage')
ret = tf.Variable(0.3, name='ret')

# state vector (as tf.Variable)
valc = tf.Variable(np.kron(np.linspace(2.0, 3.0, M).reshape((M, 1)), np.linspace(0.0, 1.0, N)), dtype=tf.float32, name='valc')
vald = tf.Variable(0.5*np.linspace(2.0, 3.0, M).reshape((M, 1)), dtype=tf.float32, name='vald')
invc = tf.Variable(np.zeros((M, N)), dtype=tf.float32, name='invc')

# total value function
defval = vald - valc
default = sigmoid(defval)
val = default*vald + (1-default)*valc
# val = maxplus(valc-vald, 0.0, sharp=10.0) + vald
# val = vald + sigmoid(valc-vald)*tf.maximum(0.0, valc-vald)
# val = tf.maximum(valc, vald)

# derivatives - use upwinding scheme from moll et al
vca_l = vderiv(val, meth='left')
vca_r = vderiv(val, meth='right')
vca = sigmoid(-invc)*vca_l + sigmoid(invc)*vca_r # this smoothness seems to promote stability

# default value function
incd = wage
consd = incd
utild = tf.log(consd)
valc0 = valc[:,nmid:nmid+1]
vdnew = dt * utild + ( 1 - dt * rho ) * ( ( 1 - ret ) * vald + ret * valc0 )

# continuation value function
incc = irate*avec + wage
consc = incc - invc
utilc = tf.log(maxplus(consc, cmin))
icnew = tf.to_float(defval < 0) * ( incc - 1 / tf.maximum(0.01, vca) )
vcnew = dt * utilc + ( 1 - dt * rho ) * ( val + dt * invc * vca )

# differentials and loss
valc_diff = vcnew - valc
vald_diff = vdnew - vald
invc_diff = icnew - invc
# loss = tf.nn.l2_loss(valc_diff) + tf.nn.l2_loss(vald_diff) + tf.nn.l2_loss(invc_diff)
loss = tf.nn.l2_loss(valc_diff) + tf.nn.l2_loss(vald_diff)

# optimizer
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# optim = opt.minimize(loss, var_list=[val, inv])

# operations
init = tf.global_variables_initializer()
vc_upd = tf.assign(valc, drag*valc + (1-drag)*vcnew)
vd_upd = tf.assign(vald, drag*vald + (1-drag)*vdnew)
ic_upd = tf.assign(invc, drag*invc + (1-drag)*icnew)
update = tf.group(vc_upd, vd_upd)
# update = tf.group(vc_upd, vd_upd, ic_upd)

# store history
cstore = [
    'loss'
]
mstore = [
    'valc', 'vald', 'invc',
    'incc', 'consc', 'utilc',
    'incd', 'consd', 'utild',
    'vca', 'vca_l', 'vca_r',
    'val', 'icnew', 'defval',
    'valc_diff', 'vald_diff', 'invc_diff'
]
store = cstore + mstore

scope = globals()
vardict = {n: scope[n] for n in store}
history = {}
history.update({n: np.zeros(K) for n in cstore})
history.update({n: np.zeros((K, M, N)) for n in mstore})

def store_hist(sess, i):
    for n in cstore:
        history[n][i] = sess.run(vardict[n])
    for n in mstore:
        history[n][i,:,:] = sess.run(vardict[n])



rvec = np.arange(K)
def plot_hist(var, rep=None, state=None, val=None):
    if state is not None and rep is not None:
        sr = rep
        si = state
        sa = slice(None)
        x = avec
    elif rep is not None:
        sr = rep
        si = slice(None)
        sa = slice(None)
        x = avec
    elif val is not None:
        sr = slice(None)
        si = slice(None)
        sa = val
        x = rvec
    else:
        x = rvec
    panes = len(var)
    nrows = ( panes - 1 ) // 3 + 1
    ncols = min(panes, 3)
    (fig, axs) = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
    for (p, ax) in zip(var, chain(*axs)):
        for n in p:
            v = eval(n, dict(np=np), history)
            if v.ndim == 1:
                ax.plot(x, v)
            elif v.ndim == 3:
                ax.plot(x, v[sr,si,sa].T)
            else:
                print('bad dim = %d' % v.ndim)
        ax.legend(p)
        sns.despine(ax=ax)

# run loops
with tf.Session() as sess:
    sess.run(init)
    for i in range(K):
        store_hist(sess, i)
        sess.run(update)

