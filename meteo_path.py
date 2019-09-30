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

class Model:
    def __init__(self, pars, vars, eqns, dtype=np.float32):
        self.pars = pars
        self.vars = vars
        self.eqns = eqns
        self.dtype = dtype

        # shape
        self.par_sh = [p.get_shape() for p in self.pars]
        self.var_sh = [v.get_shape() for v in self.vars]

        # total size
        self.par_sz = sum([prod(s) for s in self.par_sh])
        self.var_sz = sum([prod(s) for s in self.var_sh])

    def set_params(self, par):
        for p in par:
            p.assign(par[p])

    def get_params(self):
        return [p.numpy() for p in self.pars]

    def get_eqvars(self):
        return [v.numpy() for v in self.vars]

    def eqnvec(self):
        return flatify(self.eqns()).numpy()

    def parjac(self):
        return jacobian(self.eqns, self.pars).numpy()

    def varjac(self):
        return jacobian(self.eqns, self.vars).numpy()

    # solve system symbolically
    def solve_system(self, eqn_tol=1.0e-6, max_rep=20, output=False):
        if output:
            eqnvec = self.eqnvec()
            error = np.max(np.abs(eqnvec))
            print(f'error({0}) = {error}')

        for i in range(max_rep):
            # get input values
            eqnvec = self.eqnvec()
            varjac = self.varjac()

            # can be non-square so use least squares
            step = lstsq(varjac, -eqnvec)

            # updates
            diffs = unpack(step, self.var_sh)

            # operators
            upds = increment(self.vars, diffs)

            # convergence
            eqnvec = self.eqnvec()
            error = np.max(np.abs(eqnvec))

            if output:
                print(f'error({i+1}) = {error}')

            if np.isnan(error):
                if output:
                    print('OFF THE RAILS')
                return

            if error <= eqn_tol:
                break

    def homotopy_path(self, par, delt=0.01, eqn_tol=1.0e-7, max_step=1000,
                      max_newton=10, out_rep=5, solve=False, output=False):
        if solve:
            if output: print('SOLVING SYSTEM')
            self.solve_system(output=output)
            if output: print('SYSTEM SOLVED\n')

        # generate analytic homotopy paths
        par0 = self.get_params()
        var0 = self.get_eqvars()
        par1 = [self.dtype(p) for p in par]
        dp = [p1-p0 for p0, p1 in zip(par0, par1)] # assuming linear path
        dpv = np.concat(dp)

        # initalize
        t = 0.0

        if output:
            print(f't = {t}')
            eqnvec_val = self.eqnvec()
            err = np.max(np.abs(eqnvec_val))
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
            parjac_val = self.parjac()
            varjac_val = self.varjac()
            tdir_val = np.dot(parjac_val, dpv)
            fulljac_val = np.hstack([varjac_val, tdir_val[:, None]])
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
            increment(self.pars, dp, d=dt)
            increment(self.vars, dv)

            # store
            t_path.append(t)
            par_path.append(self.get_params())
            var_path.append(self.get_eqvars())

            if iout:
                print('MADE PREDICTION STEP')
                print(f't = {t}')
                eqnvec_val = self.eqnvec()
                err = np.max(np.abs(eqnvec_val))
                print(f'error = {err}')
                print()

            # correction steps
            for i in range(max_newton):
                if t == 0.0 or t == 1.0:
                    proj_dir = np.r_[np.zeros(self.var_sz), 1.0]
                else:
                    proj_dir = step_pred # project along previous step

                # get refinement step
                parjac_val = self.parjac()
                varjac_val = self.varjac()
                eqnvec_val = self.eqnvec()

                tdir_val = np.dot(parjac_val, dpv)
                fulljac_val = np.hstack([varjac_val, tdir_val[:, None]])
                projjac_val = np.vstack([fulljac_val, proj_dir])
                step_corr = -np.linalg.solve(projjac_val, np.r_[eqnvec_val, 0.0])

                # increment
                dt = step_corr[-1]
                dv = unpack(step_corr[:-1], self.var_sh)

                # implement
                t += dt
                increment(self.pars, dp, d=dt)
                increment(self.vars, dv)

                # check for convergence
                eqnvec_val = self.eqnvec()
                err = np.max(np.abs(eqnvec_val))
                if err <= eqn_tol: break

            if iout:
                print(f'MADE {i} CORRECTION STEPS')
                print(f't = {t}')
                print(f'error = {err}')
                print()

            # store
            t_path.append(t)
            par_path.append(self.get_params())
            var_path.append(self.get_eqvars())

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
