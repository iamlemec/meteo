from collections import OrderedDict
import json
import numpy as np
import theano
import theano.tensor as T

def split(vec,sizes):
  return T.split(vec,sizes,len(sizes)) if len(sizes) > 1 else [vec]

def inv(vec):
  if vec.ndim == 2:
    return np.linalg.inv(vec)
  elif vec.ndim == 0 or (vec.ndim == 1 and vec.size == 1):
      return 1.0/vec

class Model:
  def __init__(self,fname):
    with open(fname,'r') as fid:
      mod = json.load(fid,object_pairs_hook=OrderedDict)

    # parse model specification
    self.par_names = []
    self.var_names = []

    self.par_sizes = []
    self.var_sizes = []

    for (name,spec) in mod['parameters'].items():
      self.par_names += name
      if spec.get('type','scalar') == 'scalar':
        self.par_sizes.append(1)

    for (name,spec) in mod['variables'].items():
      self.var_names += name
      if spec.get('type','scalar') == 'scalar':
        self.var_sizes.append(1)

    self.n_pars = len(self.par_sizes)
    self.n_vars = len(self.var_sizes)

    # input
    self.par_vec = T.dvector('parvec')
    self.var_vec = T.dvector('varvec')

    # unpack
    self.par_dict = dict(zip(self.par_names,split(self.par_vec,self.par_sizes)))
    self.var_dict = dict(zip(self.var_names,split(self.var_vec,self.var_sizes)))
    self.sym_dict = dict(self.par_dict.items()+self.var_dict.items())

    # evaluate
    self.equations = []
    for eq in mod['equations']:
      self.equations.append(eval(eq,{},self.sym_dict))

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

  def homotopy_bde(self,par_start,par_finish,var_start,delt=0.01,eqn_tol=1.0e-8,max_step=1000,max_newton=10,plot=False):
    (par_start,par_finish,var_start) = map(np.array,(par_start,par_finish,var_start))

    path_apply = lambda t: self.path_fun(par_start,par_finish,t)
    dpath_apply = lambda t: self.dpath_fun(par_start,par_finish,t)

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

      # print out
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

      # print out
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
