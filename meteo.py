import theano
import theano.tensor as T
import json
from collections import OrderedDict

def split(vec,sizes):
  return T.split(vec,sizes,len(sizes)) if len(sizes) > 1 else [vec]

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

  def homotopy(self,par_start,par_finish,var_start,path_str='(1.0-t)*start+t*finish',delt=0.01,max_rep=1000):
    t = T.dscalar('t')
    path = eval(path_str,{},{'start':par_start,'finish':par_finish,'t':t})
    dpath = T.jacobian(path,t)
    path_fun = theano.function([t],path)
    dpath_fun = theano.function([t],dpath)

    t_path =  [0.0]
    par_path = [par_start]
    var_path = [var_start]

    tv = 0.0
    var_vec = var_start.copy()
    for rep in xrange(max_rep):
      tv += delt

      par_vec = path_fun(tv)
      dpath_vec = dpath_fun(tv)
      varjac_vec = self.varjac_fun(par_vec,var_vec)
      parjac_vec = self.parjac_fun(par_vec,var_vec)

      varijac_vec = np.linalg.inv(varjac_vec)
      vardiff_vec = -np.dot(np.dot(varijac_vec,parjac_vec),dpath_vec)
      step = delt*vardiff_vec
      var_vec += step

      print (tv,par_vec,var_vec)

      t_path.append(tv)
      par_path.append(par_vec)
      var_path.append(var_vec)

      if tv >= 1.0: break

    return (np.array(t_path),np.array(par_path),np.array(var_path))

mod = Model('model.json')
