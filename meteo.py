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

  def homotopy_newton(self,par_start,par_finish,var_start,delt=0.01,eqn_tol=1.0e-8,max_step=1000,max_newton=10):
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
    var_val = var_start.copy()
    for rep in xrange(max_step):
      # elevator increment
      deltv = np.minimum(1.0-tv,delt)
      tv += deltv

      # calculate jacobians
      par_val = path_apply(tv)
      dpath_val = dpath_apply(tv)
      varjac_val = self.varjac_fun(par_val,var_val)
      parjac_val = self.parjac_fun(par_val,var_val)

      # calculate step
      varijac_val = np.linalg.inv(varjac_val)
      vardiff_val = -np.dot(varijac_val,np.dot(parjac_val,dpath_val))
      step = deltv*vardiff_val
      var_val += step

      # new function value
      eqn_val = self.eqn_fun(par_val,var_val)

      # newton steps
      for i in xrange(max_newton):
        varjac_val = self.varjac_fun(par_val,var_val)
        varijac_val = np.linalg.inv(varjac_val)
        step = -np.dot(varijac_val,eqn_val)
        var_val += step
        eqn_val = self.eqn_fun(par_val,var_val)
        if np.max(eqn_val) <= eqn_tol: break

      # print out
      print 't = {}'.format(tv)
      print 'par_val = {}'.format(str(par_val))
      print 'var_val = {}'.format(str(var_val))
      print 'eqn_val = {}'.format(str(eqn_val))
      print

      # store
      t_path.append(tv)
      par_path.append(par_val)
      var_path.append(var_val)

      # if we can't stay on the path
      if np.max(eqn_val) > eqn_tol:
        print 'Off the rails.'
        break

      # break at end
      if tv >= 1.0: break

    return (np.array(t_path),np.array(par_path),np.array(var_path))

  def homotopy_bde(self,par_start,par_finish,var_start,delt=0.01,eqn_tol=1.0e-8,max_step=1000,max_newton=10):
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
      fulljac_val = np.hstack([varjac_val,np.dot(parjac_val,dpath_val)[:,np.newaxis]])
      jac_dets = np.array([np.linalg.det(fulljac_val.compress(np.arange(self.n_vars+1)!=i,axis=1)) for i in xrange(self.n_vars+1)])
      step = jac_dets*(-np.ones(self.n_vars+1))**np.arange(self.n_vars+1)

      # move in the right direction
      if direc is None: direc = np.sign(step[-1])
      step *= direc

      # this normalization keeps us in sane regions (equivalent to doubling equation system)
      step *= delt*np.abs(np.linalg.det(varjac_val))
      deltv = np.minimum(delt,1.0-tv)
      if step[-1] > deltv: step *= deltv/step[-1]

      # increment
      tv += step[-1]
      var_val += step[:-1]

      # new function value
      par_val = path_apply(tv)
      eqn_val = self.eqn_fun(par_val,var_val)

      # newton steps
      for i in xrange(max_newton):
        varjac_val = self.varjac_fun(par_val,var_val)
        varijac_val = np.linalg.inv(varjac_val)
        step = -np.dot(varijac_val,eqn_val)
        var_val += step
        eqn_val = self.eqn_fun(par_val,var_val)
        if np.max(eqn_val) <= eqn_tol: break

      # projection steps
      # for i in xrange(max_newton):
      #   par_val = path_apply(tv)
      #   dpath_val = dpath_apply(tv)
      #   varjac_val = self.varjac_fun(par_val,var_val)
      #   parjac_val = self.parjac_fun(par_val,var_val)
      #   fulljac_val = np.hstack([varjac_val,np.dot(parjac_val,dpath_val)[:,np.newaxis]])
      #   proj_val = np.vstack([fulljac_val,jac_dets])
      #   step = -np.dot(np.linalg.inv(proj_val),np.r_[eqn_val,0.0])
      #   tv += step[-1]
      #   var_val += step[:-1]
      #   eqn_val = self.eqn_fun(par_val,var_val)
      #   if np.max(eqn_val) <= eqn_tol: break

      # print out
      print 't = {}'.format(tv)
      print 'jac_dets = {}'.format(str(jac_dets))
      print 'par_val = {}'.format(str(par_val))
      print 'var_val = {}'.format(str(var_val))
      print 'eqn_val = {}'.format(str(eqn_val))
      print

      # store
      t_path.append(tv)
      par_path.append(par_val)
      var_path.append(var_val)

      # if we can't stay on the path
      if np.max(eqn_val) > eqn_tol:
        print 'Off the rails.'
        break

      # break at end
      if tv >= 1.0: break

    return (np.array(t_path),np.array(par_path),np.array(var_path))

# mod = Model('model.json')
