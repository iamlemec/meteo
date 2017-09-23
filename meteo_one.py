# meteo with tensorflow (python3)

# complex layout: [1r, 2r, 3r, 1i, 2i, 3i]

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

def vector_environ(vec, spec, complex=False):
    loc = 0
    vd = {}
    for (nm, sz) in spec.items():
        vd[nm] = tf.slice(vec, [loc], [sz], name=nm)
        loc += sz
    if complex:
        for (nm, sz) in spec.items():
            rv = vd[nm]
            cv = tf.slice(vec, [loc], [sz], name=nm+'_j')
            vd[nm] = tf.complex(rv, cv, name=nm+'_c')
            loc += sz
    return vd

def dict_to_array(din, spec, complex=False):
    vout = []
    for (nm, sz) in spec.items():
        val = np.real(din[nm])
        if sz == 1:
            vout += [val]
        else:
            vout += list(val)
    if complex:
        for (nm, sz) in spec.items():
            val = np.imag(din[nm])
            if sz == 1:
                vout += [val]
            else:
                vout += list(val)
    return np.array(vout)

def array_to_dict(vin, spec, complex=False):
    dout = {}
    loc = 0
    if complex:
        ts = sum(spec.values())
        vin = vin[:ts] + 1j*vin[ts:]
    for (nm, sz) in spec.items():
        val = vin[loc:loc+sz]
        if sz == 1:
            dout[nm] = val[0]
        else:
            dout[nm] = val
        loc += sz
    return dout

def gradient(eq, i, x):
    ret = tf.gradients(tf.slice(eq, [i], [0]), x)[0]
    if ret is None:
        return tf.zeros_like(x)
    else:
        return ret

def jacobian(eq, x):
    n = eq.get_shape()[0]
    return tf.transpose(tf.stack([gradient(eq, i, x) for i in range(n)], 1))

# TODO: handle non-flat inputs
class Model:
    def __init__(self, pars, vars, eqns):
        self.pars = pars
        self.vars = vars
        self.eqns = eqns

        # size
        self.par_sz = {par: int(par.get_shape()[0]) for par in pars}
        self.var_sz = {var: int(var.get_shape()[0]) for var in vars}
        self.eqn_sz = {eqn: int(eqn.get_shape()[0]) for eqn in eqns}

        # equation system and gradients
        self.eqnvec = tf.concat(eqns, 0)
        self.parjac = tf.concat([tf.concat([jacobian(eqn, x) for x in pars], 1) for eqn in eqns], 0)
        self.varjac = tf.concat([tf.concat([jacobian(eqn, x) for x in vars], 1) for eqn in eqns], 0)

        print(self.eqnvec.get_shape())
        print(self.parjac.get_shape())
        print(self.varjac.get_shape())

        # evaluation functions
        def state_evaler(f, matrix=False):
            def ev(p, v):
                y = f.eval(feed_dict=dict_merge(p, v))
                return ensure_matrix(y) if matrix else y
            return ev

        self.eqnvec_fun = state_evaler(self.eqnvec)
        self.parjac_fun = state_evaler(self.parjac, matrix=True)
        self.varjac_fun = state_evaler(self.varjac, matrix=True)

    # solve system, possibly along projection (otherwise fix t)
    def solve_system(self, p, v0, eqn_tol=1.0e-12, max_rep=20, output=False):
        v = dict_copy(v0)

        eqnvec_val = self.eqnvec_fun(p, v)
        if output:
            print('Initial error = {}'.format(np.max(np.abs(eqnvec_val))))

        for i in range(max_rep):
            varjac_val = self.varjac_fun(p, v)
            print(varjac_val.shape)
            print(eqnvec_val.shape)
            step = -np.linalg.solve(varjac_val, eqnvec_val)
            dstep = array_to_dict(step, self.var_sz)
            for k in v: v[k] += dstep[k]
            eqnvec_val = self.eqnvec_fun(p, v)

            if output:
                print('Equation error = {}'.format(np.max(np.abs(eqnvec_val))))

            if np.isnan(eqn_val).any():
                if output:
                    print('Off the rails.')
                return

            if np.max(np.abs(eqnvec_val)) <= eqn_tol: break

        return v

    def homotopy_bde(self, p0, p1, v0, delt=0.01, eqn_tol=1.0e-12,
                     max_step=1000, max_newton=10, out_rep=5,
                     solve=False, output=False, plot=False):
        # refine initial solution if needed else copy
        if solve:
            v = self.solve_system(p0, v0, output=output)
        else:
            v = dict_copy(v0)

        # generate analytic homotopy paths
        p0_vec = dict_to_array(p0, self.par_sz)
        p1_vec = dict_to_array(p1, self.par_sz)
        path_apply = lambda t: {k: (1-t)*p0[k] + t*p1[k] for k in self.pars}
        dpath_apply = lambda t: p1_vec - p0_vec

        # start path
        tv = 0.0

        if output:
            print('t = {}'.format(tv))
            #print('par_val = {}'.format(par_start))
            #print('var_val = {}'.format(var_start))
            print('Equation error = {}'.format(np.max(np.abs(self.eqnvec_fun(p0, v0)))))
            print()

        # save path
        t_path = [tv]
        par_path = [dict_copy(p0)]
        var_path = [dict_copy(v)]

        direc = None
        for rep in range(max_step):
            # calculate jacobians
            p = path_apply(tv)
            dp = dpath_apply(tv)
            varjac_val = self.varjac_fun(p, v)
            parjac_val = self.parjac_fun(p, v)

            # calculate steps
            tdir_val = np.dot(parjac_val, dp)
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
            dict_add(v, array_to_dict(step_pred[:-1], self.var_sz))

            # new function value
            p = path_apply(tv)
            eqn_val = self.eqnvec_fun(p, v)

            # store
            t_path.append(tv)
            par_path.append(dict_copy(p))
            var_path.append(dict_copy(v))

            # projection steps
            for i in range(max_newton):
                if tv == 0.0 or tv == 1.0:
                    proj_dir = np.r_[np.zeros(self.var_sz), 1.0]
                else:
                    proj_dir = step_pred # project along previous step

                dv = dpath_apply(tv)
                varjac_val = self.varjac_fun(p, v)
                parjac_val = self.parjac_fun(p, v)
                tdir_val = np.dot(parjac_val, dp)

                fulljac_val = np.hstack([varjac_val, tdir_val])
                projjac_val = np.vstack([fulljac_val, proj_dir])
                step_corr = -np.linalg.solve(projjac_val, np.r_[eqn_val, 0.0])

                tv += step_corr[-1]
                par_val = path_apply(tv)
                dict_add(v, array_to_dict(step_corr[:-1], self.var_sz))
                eqn_val = self.eqnvec_fun(p, v)

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
            par_path.append(dict_copy(p))
            var_path.append(dict_copy(v))

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

    # def homotopy_elev(self, par_start_dict, par_finish_dict, var_start_dict,
    #                   delt=0.01, eqn_tol=1.0e-12, max_step=1000, max_newton=10,
    #                   solve=False, output=False, out_rep=5, plot=False):
    #     # refine initial solution if needed
    #     if solve: var_start_dict = self.solve_system(par_start_dict, var_start_dict, output=output)
    #     if var_start_dict is None: return
    #
    #     # convert to raw arrays
    #     par_start = dict_to_array(par_start_dict, self.par_spec)
    #     par_finish = dict_to_array(par_finish_dict, self.par_spec)
    #     var_start = dict_to_array(var_start_dict, self.var_spec)
    #
    #     # generate analytic homotopy paths
    #     path_apply = lambda t: (1-t)*par_start + t*par_finish
    #     dpath_apply = lambda t: par_finish - par_start
    #
    #     # start path
    #     tv = 0.0
    #
    #     if output:
    #         print('t = {}'.format(tv))
    #         #print(' par_val = {}'.format(par_start))
    #         #print('var_val = {}'.format(var_start))
    #         print('Equation error = {}'.format(np.max(np.abs(self.eqn_fun(par_start, var_start)))))
    #         print()
    #
    #     # save path
    #     t_path = [tv]
    #     par_path = [par_start]
    #     var_path = [var_start]
    #
    #     direc = None
    #     var_val = var_start.copy()
    #     for rep in range(max_step):
    #         # calculate jacobians
    #         par_val = path_apply(tv)
    #         dpath_val = dpath_apply(tv)
    #         varjac_val = self.varjac_fun(par_val, var_val)
    #         parjac_val = self.parjac_fun(par_val, var_val)
    #
    #         # calculate steps
    #         tdir_val = parjac_val.dot(dpath_val)
    #         step_pred = -self.linsolve(varjac_val, tdir_val)
    #
    #         # increment
    #         delt1 = np.minimum(delt, 1.0-tv)
    #         tv += delt1
    #         var_val += delt1*step_pred
    #
    #         # new function value
    #         par_val = path_apply(tv)
    #         eqn_val = self.eqn_fun(par_val, var_val)
    #
    #         # store
    #         t_path.append(tv)
    #         par_path.append(par_val.copy())
    #         var_path.append(var_val.copy())
    #
    #         # correction steps
    #         for i in range(max_newton):
    #             varjac_val = self.varjac_fun(par_val, var_val)
    #             step_corr = -self.linsolve(varjac_val, eqn_val)
    #             var_val += step_corr
    #             eqn_val = self.eqn_fun(par_val, var_val)
    #             if np.max(np.abs(eqn_val)) <= eqn_tol: break
    #
    #         if output and (rep % out_rep) == 0:
    #             print('Iteration = {}'.format(rep))
    #             print('Step predict = {}'.format(step_pred[-1]))
    #             print('Correction steps = {}'.format(i))
    #             print('t = {}'.format(tv))
    #             #print('par_val = {}'.format(str(par_val)))
    #             #print('var_val = {}'.format(str(var_val)))
    #             print('Equation error = {}'.format(np.max(np.abs(eqn_val))))
    #             print()
    #
    #         # store
    #         t_path.append(tv)
    #         par_path.append(par_val.copy())
    #         var_path.append(var_val.copy())
    #
    #         # if we can't stay on the path
    #         if (np.max(np.abs(eqn_val)) > eqn_tol) or np.isnan(eqn_val).any():
    #             print('Off the rails.')
    #             break
    #
    #         # break at end
    #         if tv <= 0.0 or tv >= 1.0: break
    #
    #     (t_path, par_path, var_path) = map(np.array, (t_path, par_path, var_path))
    #
    #     if output:
    #         print('Done at {}!'.format(rep))
    #         print('t = {}'.format(tv))
    #         print('Equation error = {}'.format(np.max(np.abs(eqn_val))))
    #
    #     if plot:
    #         import matplotlib.pylab as plt
    #         plt.scatter(var_path[1::2], t_path[1::2], c='r')
    #         plt.scatter(var_path[::2], t_path[::2], c='b')
    #
    #     return (t_path, par_path, var_path)
