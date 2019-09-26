# meteo with the purest tensorflow (python3.6+)
# just elevator path without projection then gradient descent

from itertools import chain, islice
from collections import OrderedDict
from copy import deepcopy

import re
import json

import numpy as np
from scipy.sparse.linalg import spsolve
new = np.newaxis

import tensorflow as tf

# utils
def ensure_matrix(x):
    if type(x) is np.ndarray and x.ndim >= 2:
        return x
    else:
        return np.array(x, ndmin=2)

def inv(vec):
    if vec.ndim == 2:
        return np.linalg.inv(vec)
    elif vec.ndim == 0 or (vec.ndim == 1 and vec.size == 1):
            return 1.0/vec

def dict_merge(*dlist):
    return dict(chain(*[d.items() for d in dlist]))

def dict_copy(d):
    return {k: np.copy(d[k]) for k in d}

def dict_add(d0, d1):
    for k in d1:
        d0[k] += d1[k]

# HOMPACK90 STYLE
def row_dets_old(mat):
    (n, np1) = mat.shape
    qr = np.linalg.qr(mat, 'r')
    dets = np.zeros(np1)
    dets[np1-1] = 1.0
    for lw in range(2, np1+1):
            i = np1 - lw
            ik = i + 1
            dets[i] = -np.dot(qr[i, ik:], dets[ik:])/qr[i, i]
    dets *= np.sign(np.prod(np.diag(qr))) # just for the sign
    #dets *= np.prod(np.diag(qr)) # to get the scale exactly
    return dets

def row_dets(mat):
    (n, np1) = mat.shape
    q, r = tf.qr(mat)
    dets = np.zeros(np1)
    dets[np1-1] = 1.0
    for lw in range(2, np1+1):
            i = np1 - lw
            ik = i + 1
            dets[i] = -np.dot(qr[i, ik:np1], dets[ik:np1])/qr[i, i]
    dets *= np.sign(np.prod(np.diag(qr))) # just for the sign
    #dets *= np.prod(np.diag(qr)) # to get the scale exactly
    return dets

def dict_to_array(din, spec):
    vout = []
    for (nm, sz) in spec.items():
        val = np.real(din[nm])
        if sz == 1:
            vout += [val]
        else:
            vout += list(val)
    return np.array(vout)

def array_to_dict(vin, spec):
    dout = {}
    loc = 0
    for (nm, sz) in spec.items():
        val = vin[loc:loc+sz]
        if sz == 1:
            dout[nm] = val[0]
        else:
            dout[nm] = val
        loc += sz
    return dout

def gradient(eq, i, x):
    ret = tf.gradients(tf.slice(eq, [i], [1]), x)[0]
    if ret is None:
        return tf.zeros_like(x)
    else:
        return ret

def jacobian(eq, x):
    n = eq.get_shape()[0]
    return tf.stack([gradient(eq, i, x) for i in range(n)], 0)

def varname(nm):
    #ret = re.match(r'(.+?)(_[0-9]*)(:[0-9]*)', nm)
    #return ret.group(1) if ret is not None else nm
    if ':' in nm: nm = ''.join(nm.split(':')[:-1])
    if '_' in nm: nm = ''.join(nm.split('_')[:-1])
    return nm

def resolve(d):
    return {varname(v.name): d[v] for v in d}

# TODO: handle non-flat inputs
class Model:
    def __init__(self, pars, vars, eqns, learning_rate=0.1):
        self.pars = pars
        self.vars = vars
        self.eqns = eqns

        # size
        self.par_sz = {par: int(par.get_shape()[0]) for par in pars}
        self.var_sz = {var: int(var.get_shape()[0]) for var in vars}
        self.eqn_sz = {eqn: int(eqn.get_shape()[0]) for eqn in eqns}

        # equation system
        self.eqnvec = tf.concat(eqns, 0)
        self.error = tf.reduce_max(tf.abs(self.eqnvec))

        # gradient descent
        self.loss = tf.nn.l2_loss(self.eqnvec)
        self.gdesc_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.gdesc_update = self.gdesc_opt.minimize(self.loss, var_list=self.vars)

        # homotopy
        self.tv = tf.placeholder(dtype=tf.float64)
        self.par0 = [tf.Variable(np.zeros(p.shape)) for p in pars]
        self.par1 = [tf.Variable(np.zeros(p.shape)) for p in pars]

        # path gen
        self.path_assign = tf.group(*[p.assign((1-self.tv)*p0 + self.tv*p1)
            for p, p0, p1 in zip(pars, self.par0, self.par1)])

    # solve system symbolically
    def solve_system(self, eqn_tol=1.0e-12, max_rep=500, output=False):
        if output:
            error = self.error.eval()
            print(f'error({0}) = {error}')

        for i in range(max_rep):
            self.gdesc_update.run()
            error = self.error.eval()

            if output:
                print(f'error({i+1}) = {error}')

            if np.isnan(error):
                if output:
                    print('OFF THE RAILS')
                return

            if error <= eqn_tol:
                break

        return i + 1

    def homotopy_elev(self, par1, delt=0.01, eqn_tol=1.0e-12, max_step=1000,
                          max_newton=500, out_rep=5, solve=True, output=False,
                          newton_output=False, plot=False):
        if solve:
            print('SOLVING SYSTEM')
            self.solve_system(output=newton_output)

        # start path
        tv = 0.0
        for p, p0, p1 in zip(self.pars, self.par0, self.par1):
            p0.assign(p).eval()
            p1.assign(par1[p]).eval()

        if output:
            print(f't = {tv}')
            print(f'error = {self.error.eval()}')
            print()

        # save path
        t_path = [tv]
        par_path = [{p: p.eval() for p in self.pars}]
        var_path = [{v: v.eval() for v in self.vars}]

        direc = None
        for rep in range(max_step):
            iout = output and (rep % out_rep) == 0
            if iout:
                print(f'ITERATION = {rep}')
                print()

            # increment
            tv += np.minimum(delt, 1.0-tv)
            self.path_assign.run(feed_dict={self.tv: tv})

            # store
            t_path.append(tv)
            par_path.append({p: p.eval() for p in self.pars})
            var_path.append({v: v.eval() for v in self.vars})

            if iout:
                print('MADE PREDICTION STEP')
                print(f't = {tv}')
                print(f'error = {self.error.eval()}')
                print()

            # resolve system
            i = self.solve_system(output=newton_output, max_rep=max_newton, eqn_tol=eqn_tol)

            if iout:
                print(f'MADE {i} CORRECTION STEPS')
                print(f'error = {self.error.eval()}')
                print()

            # store
            t_path.append(tv)
            par_path.append({p: p.eval() for p in self.pars})
            var_path.append({v: v.eval() for v in self.vars})

            # if we can't stay on the path
            error = self.error.eval()
            if (error > eqn_tol) or np.isnan(error):
                print('OFF THE RAILS')
                break

            # break at end
            if tv <= 0.0 or tv >= 1.0:
                break

        if output:
            print(f'DONE AT {rep}!')
            print(f't = {tv}')
            print(f'error = {self.error.eval()}')

        if plot:
            import matplotlib.pylab as plt
            plt.scatter(var_path[1::2], t_path[1::2], c='r')
            plt.scatter(var_path[::2], t_path[::2], c='b')

        return (t_path, par_path, var_path)
