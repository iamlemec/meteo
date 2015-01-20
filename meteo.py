from itertools import chain
from collections import OrderedDict
import json
import numpy as np
import theano
import theano.tensor as T

# utils
def split(vec,sizes):
  return T.split(vec,sizes,len(sizes)) if len(sizes) > 1 else [vec]

def inv(vec):
  if vec.ndim == 2:
    return np.linalg.inv(vec)
  elif vec.ndim == 0 or (vec.ndim == 1 and vec.size == 1):
      return 1.0/vec

def merge(*dlist):
  return dict(chain(*[d.items() for d in dlist]))

class Model:
  def __init__(self,fname):
    # parse model specification
    with open(fname,'r') as fid:
      mod = json.load(fid,object_pairs_hook=OrderedDict)

    # parameters
    self.par_info = OrderedDict()
    self.par_sizes = []
    for (name,spec) in mod['parameters'].items():
      ptype = spec.get('type','scalar')
      psize = 1 if ptype == 'scalar' else spec['size']

      info = OrderedDict()
      info['type'] = ptype
      info['size'] = psize

      self.par_info[name] = info
      self.par_sizes.append(psize)

    # variables
    self.var_info = OrderedDict()
    self.var_sizes = []
    for (name,spec) in mod['variables'].items():
      vtype = spec.get('type','scalar')
      vsize = 1 if vtype == 'scalar' else spec['size']

      info = OrderedDict()
      info['type'] = vtype
      info['size'] = vsize
      if vtype == 'function':
        vder = spec.get('derivatives',0)
        (tmin,tmax) = spec['range']
        (tbound,fbound) = spec['boundary']
        info['nder'] = vder
        info['min'] = tmin
        info['max'] = tmax
        info['tbound'] = tbound
        info['fbound'] = fbound
      else:
        vder = 0

      self.var_info[name] = info
      self.var_sizes.append(vsize)
      if vder > 0: self.var_sizes += vder*[vsize]

    # totals
    self.n_pars = len(self.par_info)
    self.n_vars = len(self.var_info)

    # input vectors
    self.par_vec = T.dvector('parvec')
    self.var_vec = T.dvector('varvec')

    # unpack and map out variables
    self.par_dict = OrderedDict()
    piter = iter(split(self.par_vec,self.par_sizes))
    for (name,info) in self.par_info.items():
      self.par_dict[name] = piter.next()

    self.var_dict = OrderedDict()
    self.der_dict = OrderedDict()
    viter = iter(split(self.var_vec,self.var_sizes))
    for (name,info) in self.var_info.items():
      var = viter.next()
      self.var_dict[name] = var
      if info['type'] == 'function':
        self.der_dict[var] = [var] + [viter.next() for d in xrange(info['nder'])]

    # define derivative operator
    def diff(var,n=1):
      return self.der_dict[var][n]
    self.diff_dict = {'diff':diff}

    # combine them all
    self.sym_dict = merge(self.par_dict,self.var_dict,self.diff_dict)

    # evaluate
    self.equations = []

    # regular equations
    for eq in mod['equations']:
      self.equations.append(eval(eq,{},self.sym_dict))

    # derivative relations and boundary condition
    for (name,info) in self.var_info.items():
      if info['type'] == 'function':
        var = self.var_dict[name]
        size = info['size']
        (tmin,tmax) = (info['min'],info['max'])
        (tbound,fbound) = (info['tbound'],info['fbound'])

        grid = np.linspace(tmin,tmax,size)
        tint = (tbound<=grid).nonzero()[0][-1]
        tfrac = tbound - grid[tint]

        vbound = (1.0-tfrac)*var[tint:tint+1] + tfrac*var[tint+1:tint+2]
        self.equations.append(vbound-fbound)

        for d in xrange(info['nder']):
          d0 = diff(var,d)
          d1 = diff(var,d+1)
          self.equations.append(d0[1:]-d0[:-1]-grid[:-1]*d1[:-1])

    # repack
    self.eqn_vec = T.join(0,*self.equations)

    # jacobians
    self.par_jac = T.jacobian(self.eqn_vec,self.par_vec)
    self.var_jac = T.jacobian(self.eqn_vec,self.var_vec)

    # compile
    print 'Compiling...'
    self.eqn_fun = theano.function([self.par_vec,self.var_vec],self.eqn_vec)
    self.parjac_fun = theano.function([self.par_vec,self.var_vec],self.par_jac)
    self.varjac_fun = theano.function([self.par_vec,self.var_vec],self.var_jac)

    # newtonian path
    t = T.dscalar('t')
    start = T.dvector('start')
    finish = T.dvector('finish')
    path = (1.0-t)*start + t*finish
    dpath = T.jacobian(path,t)
    self.path_fun = theano.function([start,finish,t],path)
    self.dpath_fun = theano.function([start,finish,t],dpath)

  def homotopy_bde(self,par_start,par_finish,var_start,delt=0.01,eqn_tol=1.0e-8,max_step=1000,max_newton=10,output=False,plot=False):
    (par_start,par_finish,var_start) = map(np.array,(par_start,par_finish,var_start))

    path_apply = lambda t: self.path_fun(par_start,par_finish,t)
    dpath_apply = lambda t: self.dpath_fun(par_start,par_finish,t)

    if output:
      print 't = 0.0'
      print 'par_val = {}'.format(par_start)
      print 'var_val = {}'.format(var_start)
      print 'eqn_val = {}'.format(str(self.eqn_fun(par_start,var_start)))
      print

    t_path = [0.0]
    par_path = [par_start]
    var_path = [var_start]

    tv = 0.0
    direc = None
    var_val = var_start.copy()
    for rep in xrange(max_step):
      # calculate jacobians
      par_val = path_apply(tv)
      dpath_val = dpath_apply(tv)
      varjac_val = self.varjac_fun(par_val,var_val)
      parjac_val = self.parjac_fun(par_val,var_val)

      # calculate steps
      tdir_val = np.dot(parjac_val,dpath_val)[:,np.newaxis]
      fulljac_val = np.hstack([varjac_val,tdir_val])
      jac_dets = np.array([np.linalg.det(fulljac_val.compress(np.arange(self.n_vars+1)!=i,axis=1)) for i in xrange(self.n_vars+1)])
      step_pred = jac_dets*(-np.ones(self.n_vars+1))**np.arange(self.n_vars+1)

      # move in the right direction
      if direc is None: direc = np.sign(step_pred[-1])
      step_pred *= direc

      # this normalization keeps us in sane regions
      step_pred *= delt
      #step_pred *= np.abs(np.linalg.det(varjac_val))

      # bound between [0,1] and limit step size
      delt_max = np.minimum(delt,1.0-tv)
      delt_min = np.maximum(-delt,-tv)
      if step_pred[-1] > delt_max: step_pred *= delt_max/step_pred[-1]
      if step_pred[-1] < delt_min: step_pred *= delt_min/step_pred[-1]

      # increment
      tv += step_pred[-1]
      var_val += step_pred[:-1]

      # new function value
      par_val = path_apply(tv)
      eqn_val = self.eqn_fun(par_val,var_val)

      if output:
        print 'Predictor Step ({})'.format(rep)
        print 't = {}'.format(tv)
        print 'par_val = {}'.format(str(par_val))
        print 'var_val = {}'.format(str(var_val))
        print 'eqn_val = {}'.format(str(eqn_val))
        print

      # store
      t_path.append(tv)
      par_path.append(par_val.copy())
      var_path.append(var_val.copy())

      # projection steps
      tv0 = tv
      var_val0 = var_val.copy()
      for i in xrange(max_newton):
        if tv == 0.0 or tv == 1.0:
          proj_dir = np.r_[np.zeros(self.n_vars),1.0]
        else:
          proj_dir = step_pred # project along previous step

        dpath_val = dpath_apply(tv)
        varjac_val = self.varjac_fun(par_val,var_val)
        parjac_val = self.parjac_fun(par_val,var_val)
        tdir_val = np.dot(parjac_val,dpath_val)[:,np.newaxis]

        fulljac_val = np.hstack([varjac_val,tdir_val])
        projjac_val = np.vstack([fulljac_val,proj_dir])
        step_corr = -np.dot(np.linalg.inv(projjac_val),np.r_[eqn_val,0.0])

        tv += step_corr[-1]
        par_val = path_apply(tv)
        var_val += step_corr[:-1]
        eqn_val = self.eqn_fun(par_val,var_val)

        if np.max(np.abs(eqn_val)) <= eqn_tol: break

      if output:
        print 'Corrector Step ({})'.format(i)
        print 't = {}'.format(tv)
        print 'par_val = {}'.format(str(par_val))
        print 'var_val = {}'.format(str(var_val))
        print 'eqn_val = {}'.format(str(eqn_val))
        print

      # store
      t_path.append(tv)
      par_path.append(par_val.copy())
      var_path.append(var_val.copy())

      # if we can't stay on the path
      # if np.max(np.abs(eqn_val)) > eqn_tol:
      #   print 'Off the rails.'
      #   break

      # break at end
      if tv <= 0.0 or tv >= 1.0: break

    (t_path,par_path,var_path) = map(np.array,(t_path,par_path,var_path))

    if plot:
      import matplotlib.pylab as plt
      plt.scatter(var_path[1::2],t_path[1::2],c='r')
      plt.scatter(var_path[::2],t_path[::2],c='b')

    return (t_path,par_path,var_path)

# mod = Model('model.json')
# (t_path,par_path,var_path) = mod.homotopy_bde([2.0],[1.0],[1.0])
# (t_path,par_path,var_path) = mod.homotopy_bde([2.0],[3.0],[1.0])
