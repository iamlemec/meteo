# meteo with the purest tensorflow (python3.6+)

import re
import json
from itertools import chain, islice
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import tensorflow as tf
from scipy.sparse.linalg import spsolve
from tools import *

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

def rename(x, name):
    return tf.identity(x, name=name)

def summary(d):
    t = type(d)
    if t is dict:
        return {k.name: v for k, v in d.items()}

# HOM-HOM-HOM-HOM HOMPACK90 STYLE
def row_dets(mat):
    n, np1 = mat.shape
    qr = np.linalg.qr(mat, 'r')
    dets = np.zeros(np1, dtype=mat.dtype)
    dets[np1-1] = 1.0
    for lw in range(2, np1+1):
        i = np1 - lw
        ik = i + 1
        dets[i] = -np.dot(qr[i, ik:np1], dets[ik:np1])/qr[i, i]
    dets *= np.sign(np.prod(np.diag(qr))) # just for the sign
    # dets *= np.prod(np.diag(qr)) # to get the scale exactly
    return dets

def varname(nm):
    if ':' in nm: nm = ''.join(nm.split(':')[:-1])
    if '_' in nm: nm = ''.join(nm.split('_')[:-1])
    return nm

def resolve(d):
    return {varname(v.name): d[v] for v in d}

# TODO: handle non-flat inputs
class Model:
    def __init__(self, pars, vars, eqns, dtype=np.float32):
        self.pars = pars
        self.vars = vars
        self.eqns = eqns
        self.dtype = dtype

        # shape
        self.par_sh = [p.get_shape() for p in self.pars]
        self.var_sh = [v.get_shape() for v in self.vars]
        self.eqn_sh = [e.get_shape() for e in self.eqns]

        # total size
        self.par_sz = sum([prod(s) for s in self.par_sh])
        self.var_sz = sum([prod(s) for s in self.var_sh])
        self.eqn_sz = sum([prod(s) for s in self.eqn_sh])

        # equation system
        self.parvec = flatify(self.pars)
        self.varvec = flatify(self.vars)
        self.eqnvec = flatify(self.eqns)
        self.error = tf.reduce_max(tf.abs(self.eqnvec))

        # gradients
        self.parjac = jacobian(self.eqnvec, self.pars)
        self.varjac = jacobian(self.eqnvec, self.vars)

        # newton steps
        self.newton = newton_step(self.eqnvec, self.vars, jac=self.varjac)

        # target param
        self.par0 = [tf.zeros_like(p) for p in self.pars]
        self.par1 = [tf.zeros_like(p) for p in self.pars]
        self.t = tf.Variable(0.0, name='dt')
        self.path = [self.t*p1 + (1-self.t)*p0 for p0, p1 in zip(self.par0, self.par1)]
        self.update_path = assign(self.pars, self.path)

    def set_params(self, pars, sess=None):
        par_list = [pars[p] for p in self.pars]
        sess.run(assign(self.pars, par_list))

    def eval_system(self, sess=None):
        return {eq: sess.run(eq) for eq in self.eqns}

    # solve system symbolically
    def solve_system(self, eqn_tol=1.0e-7, max_rep=20, output=False, sess=None):
        if output:
            error = self.error.eval()
            print(f'error({0}) = {error}')

        for i in range(max_rep):
            sess.run(self.newton)
            error = self.error.eval()

            if output:
                print(f'error({i+1}) = {error}')

            if np.isnan(error):
                if output:
                    print('OFF THE RAILS')
                return

            if error <= eqn_tol:
                break

    def homotopy_path(self, par, delt=0.01, eqn_tol=1.0e-7, max_step=1000,
                      max_newton=10, out_rep=5, solve=False, output=False,
                      plot=False, sess=None):
        if solve:
            if output: print('SOLVING SYSTEM')
            self.solve_system(output=output)

        # generate analytic homotopy paths
        par0 = sess.run(self.pars)
        var0 = sess.run(self.vars)
        par1 = [self.dtype(par[p]) for p in self.pars]
        dp = np.concat([(p1-p0).flatten() for p0, p1 in zip(par0, par1)]) # assuming linear path

        # initalize
        t = 0.0

        if output:
            print(f't = {t}')
            err = self.error.eval()
            print(f'error = {err}')
            print()

        # save path
        t_path = [t]
        par_path = [par0]
        var_path = [var0]

        direc = None
        for rep in range(max_step):
            iout = output and (rep % out_rep) == 0
            if iout:
                print(f'ITERATION = {rep}')
                print()

            # prediction step
            parjac_val = sess.run(self.parjac)
            varjac_val = sess.run(self.varjac)
            tdir_val = np.dot(parjac_val, dp)[:, None]
            fulljac_val = np.hstack([varjac_val, tdir_val])
            step_pred = row_dets(fulljac_val)

            if np.mean(np.abs(step_pred)) == 0.0:
                # elevator step
                step_pred[:] = np.zeros_like(step_pred)
                step_pred[-1] = np.minimum(delt, 1.0-t)
                direc = None
            else:
                # move in the right direction
                if direc is None: direc = np.sign(step_pred[-1])
                step_pred *= direc

                # this normalization keeps us in sane regions
                #step_pred *= delt
                step_pred *= delt/np.mean(np.abs(step_pred))

                # bound between [0,1] and limit step size
                delt_max = np.minimum(delt, 1.0-t)
                delt_min = np.maximum(-delt, -t)
                if step_pred[-1] > delt_max: step_pred *= delt_max/step_pred[-1]
                if step_pred[-1] < delt_min: step_pred *= delt_min/step_pred[-1]

            # increment
            dt = step_pred[-1]
            dv = unpack(step_pred[:-1], self.var_sh)

            # implement
            t += dt
            sess.run(increment(self.pars, dp, d=dt))
            sess.run(increment(self.vars, dv))

            # store
            t_path.append(t)
            par_path.append(sess.run(self.pars))
            var_path.append(sess.run(self.vars))

            if iout:
                print('MADE PREDICTION STEP')
                print(f't = {t}')
                err = self.error.eval()
                print(f'error = {err}')
                print()

            # correction steps
            for i in range(max_newton):
                if t == 0.0 or t == 1.0:
                    proj_dir = np.r_[np.zeros(self.var_sz), 1.0]
                else:
                    proj_dir = step_pred # project along previous step

                # get refinement step
                parjac_val = sess.run(self.parjac)
                varjac_val = sess.run(self.varjac)
                eqnvec_val = sess.run(self.eqnvec)

                tdir_val = np.dot(parjac_val, dp)[:, None]
                fulljac_val = np.hstack([varjac_val, tdir_val])
                projjac_val = np.vstack([fulljac_val, proj_dir])
                step_corr = -np.linalg.solve(projjac_val, np.r_[eqnvec_val, 0.0])

                # increment
                dt = step_corr[-1]
                dv = unpack(step_corr[:-1], self.var_sh)

                # implement
                t += dt
                sess.run(increment(self.pars, dp, d=dt))
                sess.run(increment(self.vars, dv))

                # check for convergence
                err = self.error.eval()
                if err <= eqn_tol: break

            if iout:
                print(f'MADE {i} CORRECTION STEPS')
                print(f't = {t}')
                print(f'error = {err}')
                print()

            # store
            t_path.append(t)
            par_path.append(sess.run(self.pars))
            var_path.append(sess.run(self.vars))

            # if we can't stay on the path
            if (err > eqn_tol) or np.isnan(eqnvec_val).any():
                print('OFF THE RAILS')
                break

            # break at end
            if t <= 0.0 or t >= 1.0: break

        if output:
            print(f'DONE AT {rep}!')
            print(f't = {t}')
            print(f'error = {err}')

        return t_path, par_path, var_path
