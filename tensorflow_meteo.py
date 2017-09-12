# meteo with tensorflow (python3)

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

def merge(*dlist):
    return dict(chain.from_iterable([d.items() for d in dlist]))

# HOMPACK90 STYLE
def row_dets(mat):
    (n,np1) = mat.shape
    qr = np.linalg.qr(mat, 'r')
    dets = np.zeros(np1)
    dets[np1-1] = 1.0
    for lw in range(2, np1+1):
            i = np1 - lw
            ik = i + 1
            dets[i] = -np.dot(qr[i, ik:np1], dets[ik:np1])/qr[i, i]
    dets *= np.sign(np.prod(np.diag(qr))) # just for the sign
    #dets *= np.prod(np.diag(qr)) # to get the scale exactly
    return dets

def vector_environ(vec, spec):
    loc = 0
    vd = {}
    for (nm, sz) in spec.items():
        vd[nm] = tf.slice(vec, [loc], [sz], name=nm)
        loc += sz
    return vd

def dict_to_array(self, din, spec):
    vout = []
    for (nm, sz) in spec.items():
        val = din[nm]
        if sz == 1:
            vout += [val]
        else:
            vout += list(val)
    return np.array(vout)

def array_to_dict(self, vin, spec):
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

class Model:
    def __init__(self, model, zfuzz=None):
        spec = json.load(open(model)) if type(model) is str else model
        self.const = spec.get('constants', {})
        self.par_spec = spec['parameters']
        self.var_spec = spec['variables']
        self.complex = self.zfuzz is not None
        self.zfuzz = zfuzz

        # fill in defaults
        self.sz_pars = sum(self.par_spec.values())
        self.sz_vars = sum(self.var_spec.values())

        # input vectors
        self.par_vec = tf.Variable(np.zeros(self.sz_pars, dtype=tf.float64), name='parvec')
        if self.complex:
            self.cvar_vec = tf.Variable(np.zeros(self.sz_vars, dtype=tf.complex128, name='varvec')
            self.var_vec = tf.concat([tf.real(self.cvar_vec), tf.imag(self.cvar_vec)], 0)
        else:
            self.var_vec = tf.Variable(np.zeros(self.sz_vars, dtype=tf.float64), name='varvec')

        # environments
        self.par_env = vector_environ(self.par_vec, self.par_spec)
        self.var_env = vector_environ(self.var_vec, self.var_spec)
        self.env = merge(self.const, self.par_env, self.var_env)

        # parse equations
        if 'equations' in spec: self.generate_system(spec['equations'])

        # solver
        self.linsolve = np.linalg.solve

    def generate_system(self, eqs):
        resolve = lambda eq: eval(eq, self.env) if type(eq) is str else eq
        self.eqn_exp = {nm: resolve(eq) for (nm, eq) in eqs.items()}

        # equations system
        self.eqn_vec = tf.concat(list(self.eqn_exp.values()), 0)
        self.eqn_spec = {nm: int(eq.get_shape()[0]) for (nm, eq) in self.eqn_exp.items()}
        self.eqn_sz = sum(self.eqn_spec.values())

        # find gradients
        n_eqns = self.eqn_vec.get_shape()[0]
        eqn_list = tf.split(self.eqn_vec, n_eqns)
        self.par_jac = [tf.gradients(eqn, self.par_vec)[0] for eqn in eqn_list]
        self.var_jac = [tf.gradients(eqn, self.var_vec)[0] for eqn in eqn_list]

        # create functions
        def state_evaler(f):
            if type(f) is list:
                def ev(p,v):
                    env = {self.par_vec: p, self.var_vec: v}
                    return np.array([fi.eval(env) for fi in f])
            else:
                def ev(p, v):
                    env = {self.par_vec: p, self.var_vec: v}
                    return f.eval(env)
            return ev

        self.eqn_fun = state_evaler(self.eqn_vec)
        self.parjac_fun = state_evaler(self.par_jac)
        self.varjac_fun = state_evaler(self.var_jac)

    def eval_system(self, par_dict, var_dict, output=False):
        par_val = dict_to_array(par_dict, self.par_spec)
        var_val = dict_to_array(var_dict, self.var_spec)
        eqn_val = self.eqn_fun(par_val, var_val)

        if output:
            print('par_val = {}'.format(str(par_val)))
            print('var_val = {}'.format(str(var_val)))
            print('eqn_val = {}'.format(str(eqn_val)))
            print()

        return array_to_dict(eqn_val, self.eqn_spec)

    # solve system, possibly along projection (otherwise fix t)
    def solve_system(self, par_dict, var_dict, eqn_tol=1.0e-12, max_rep=20, output=False):
        par_val = dict_to_array(par_dict, self.par_spec)
        var_val = dict_to_array(var_dict, self.var_spec)
        eqn_val = self.eqn_fun(par_val, var_val)

        if output:
            print('Initial error = {}'.format(np.max(np.abs(eqn_val))))

        for i in range(max_rep):
            varjac_val = ensure_matrix(self.varjac_fun(par_val, var_val))
            step = -self.linsolve(varjac_val, eqn_val)
            var_val += step
            eqn_val = self.eqn_fun(par_val, var_val)

            if np.isnan(eqn_val).any():
                if output:
                    print('Off the rails.')
                return

            if np.max(np.abs(eqn_val)) <= eqn_tol: break

        if output:
            print('Equation Solved ({})'.format(i))
            print('par_val = {}'.format(str(par_val)))
            print('var_val = {}'.format(str(var_val)))
            print('eqn_val = {}'.format(str(eqn_val)))
            print()

        return array_to_dict(var_val, self.var_spec)

    def homotopy_bde(self, par_start_dict, par_finish_dict, var_start_dict,
                     delt=0.01, eqn_tol=1.0e-12, max_step=1000, max_newton=10,
                     solve=False, output=False, out_rep=5, plot=False):
        # refine initial solution if needed
        if solve: var_start_dict = self.solve_system(par_start_dict, var_start_dict, output=output)

        # convert to raw arrays
        par_start = dict_to_array(par_start_dict, self.par_spec)
        par_finish = dict_to_array(par_finish_dict, self.par_spec)
        var_start = dict_to_array(var_start_dict, self.var_spec)

        # generate analytic homotopy paths
        path_apply = lambda t: (1-t)*par_start + t*par_finish
        dpath_apply = lambda t: par_finish - par_start

        # start path
        tv = 0.0

        if output:
            print('t = {}'.format(tv))
            #print('par_val = {}'.format(par_start))
            #print('var_val = {}'.format(var_start))
            print('Equation error = {}'.format(np.max(np.abs(self.eqn_fun(par_start, var_start)))))
            print()

        # save path
        t_path = [tv]
        par_path = [par_start]
        var_path = [var_start]

        direc = None
        var_val = var_start.copy()
        for rep in range(max_step):
            # calculate jacobians
            par_val = path_apply(tv)
            dpath_val = dpath_apply(tv)
            varjac_val = self.varjac_fun(par_val, var_val)
            parjac_val = self.parjac_fun(par_val, var_val)

            # calculate steps
            tdir_val = np.dot(parjac_val, dpath_val)[:, new]
            fulljac_val = np.hstack([varjac_val, tdir_val])
            step_pred = row_dets(fulljac_val)

            if np.mean(np.abs(step_pred)) == 0.0:
                # elevator step
                step_pred[:] = np.zeros_like(step_pred)
                step_pred[-1] = np.minimum(delt, 1.0-tv)
                direc = None
            else:
                # move in the right direction
                if direc is None: direc = np.sign(step_pred[-1])
                step_pred *= direc

                # this normalization keeps us in sane regions
                #step_pred *= delt
                step_pred *= delt/np.mean(np.abs(step_pred))

                # bound between [0,1] and limit step size
                delt_max = np.minimum(delt, 1.0-tv)
                delt_min = np.maximum(-delt, -tv)
                if step_pred[-1] > delt_max: step_pred *= delt_max/step_pred[-1]
                if step_pred[-1] < delt_min: step_pred *= delt_min/step_pred[-1]

            # increment
            tv += step_pred[-1]
            var_val += step_pred[:-1]

            # new function value
            par_val = path_apply(tv)
            eqn_val = self.eqn_fun(par_val, var_val)

            # store
            t_path.append(tv)
            par_path.append(par_val.copy())
            var_path.append(var_val.copy())

            # projection steps
            tv0 = tv
            var_val0 = var_val.copy()
            for i in range(max_newton):
                if tv == 0.0 or tv == 1.0:
                    proj_dir = np.r_[np.zeros(self.sz_vars), 1.0]
                else:
                    proj_dir = step_pred # project along previous step

                dpath_val = dpath_apply(tv)
                varjac_val = self.varjac_fun(par_val, var_val)
                parjac_val = self.parjac_fun(par_val, var_val)
                tdir_val = np.dot(parjac_val, dpath_val)[:, new]

                fulljac_val = np.hstack([varjac_val, tdir_val])
                projjac_val = np.vstack([fulljac_val, proj_dir])
                step_corr = -np.linalg.solve(projjac_val, np.r_[eqn_val, 0.0])

                tv += step_corr[-1]
                par_val = path_apply(tv)
                var_val += step_corr[:-1]
                eqn_val = self.eqn_fun(par_val, var_val)

                if np.max(np.abs(eqn_val)) <= eqn_tol: break

            if output and (rep % out_rep) == 0:
                print('Iteration = {}'.format(rep))
                print('Step predict = {}'.format(step_pred[-1]))
                print('Correction steps = {}'.format(i))
                print('t = {}'.format(tv))
                #print('par_val = {}'.format(str(par_val)))
                #print('var_val = {}'.format(str(var_val)))
                print('Equation error = {}'.format(np.max(np.abs(eqn_val))))
                print()

            # store
            t_path.append(tv)
            par_path.append(par_val.copy())
            var_path.append(var_val.copy())

            # if we can't stay on the path
            if (np.max(np.abs(eqn_val)) > eqn_tol) or np.isnan(eqn_val).any():
                print('Off the rails.')
                break

            # break at end
            if tv <= 0.0 or tv >= 1.0: break

        (t_path, par_path, var_path) = map(np.array, (t_path, par_path, var_path))

        if output:
            print('Done at {}!'.format(rep))
            print('t = {}'.format(tv))
            print('Equation error = {}'.format(np.max(np.abs(eqn_val))))

        if plot:
            import matplotlib.pylab as plt
            plt.scatter(var_path[1::2], t_path[1::2], c='r')
            plt.scatter(var_path[::2], t_path[::2], c='b')

        return (t_path, par_path, var_path)

    def homotopy_elev(self, par_start_dict, par_finish_dict, var_start_dict,
                      delt=0.01, eqn_tol=1.0e-12, max_step=1000, max_newton=10,
                      solve=False, output=False, out_rep=5, plot=False):
        # refine initial solution if needed
        if solve: var_start_dict = self.solve_system(par_start_dict, var_start_dict, output=output)
        if var_start_dict is None: return

        # convert to raw arrays
        par_start = dict_to_array(par_start_dict, self.par_spec)
        par_finish = dict_to_array(par_finish_dict, self.par_spec)
        var_start = dict_to_array(var_start_dict, self.var_spec)

        # generate analytic homotopy paths
        path_apply = lambda t: (1-t)*par_start + t*par_finish
        dpath_apply = lambda t: par_finish - par_start

        # start path
        tv = 0.0

        if output:
            print('t = {}'.format(tv))
            #print(' par_val = {}'.format(par_start))
            #print('var_val = {}'.format(var_start))
            print('Equation error = {}'.format(np.max(np.abs(self.eqn_fun(par_start, var_start)))))
            print()

        # save path
        t_path = [tv]
        par_path = [par_start]
        var_path = [var_start]

        direc = None
        var_val = var_start.copy()
        for rep in range(max_step):
            # calculate jacobians
            par_val = path_apply(tv)
            dpath_val = dpath_apply(tv)
            varjac_val = self.varjac_fun(par_val, var_val)
            parjac_val = self.parjac_fun(par_val, var_val)

            # calculate steps
            tdir_val = parjac_val.dot(dpath_val)
            step_pred = -self.linsolve(varjac_val, tdir_val)

            # increment
            delt1 = np.minimum(delt, 1.0-tv)
            tv += delt1
            var_val += delt1*step_pred

            # new function value
            par_val = path_apply(tv)
            eqn_val = self.eqn_fun(par_val, var_val)

            # store
            t_path.append(tv)
            par_path.append(par_val.copy())
            var_path.append(var_val.copy())

            # correction steps
            for i in range(max_newton):
                varjac_val = ensure_matrix(self.varjac_fun(par_val, var_val))
                step_corr = -self.linsolve(varjac_val, eqn_val)
                var_val += step_corr
                eqn_val = self.eqn_fun(par_val, var_val)
                if np.max(np.abs(eqn_val)) <= eqn_tol: break

            if output and (rep % out_rep) == 0:
                print('Iteration = {}'.format(rep))
                print('Step predict = {}'.format(step_pred[-1]))
                print('Correction steps = {}'.format(i))
                print('t = {}'.format(tv))
                #print('par_val = {}'.format(str(par_val)))
                #print('var_val = {}'.format(str(var_val)))
                print('Equation error = {}'.format(np.max(np.abs(eqn_val))))
                print()

            # store
            t_path.append(tv)
            par_path.append(par_val.copy())
            var_path.append(var_val.copy())

            # if we can't stay on the path
            if (np.max(np.abs(eqn_val)) > eqn_tol) or np.isnan(eqn_val).any():
                print('Off the rails.')
                break

            # break at end
            if tv <= 0.0 or tv >= 1.0: break

        (t_path, par_path, var_path) = map(np.array, (t_path, par_path, var_path))

        if output:
            print('Done at {}!'.format(rep))
            print('t = {}'.format(tv))
            print('Equation error = {}'.format(np.max(np.abs(eqn_val))))

        if plot:
            import matplotlib.pylab as plt
            plt.scatter(var_path[1::2], t_path[1::2], c='r')
            plt.scatter(var_path[::2], t_path[::2], c='b')

        return (t_path, par_path, var_path)
