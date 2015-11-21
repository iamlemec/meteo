# meteo with tensorflow (python2)

import tensorflow as tf
import numpy as np

# alg params
N = 100 + 1
kmin = 0.1
kmax = 2.0
delt = 0.1

# mod params
alpha = 0.3
beta = 0.05
delta = 0.1

# steady state
kss = (alpha/beta)**(1/(1-alpha))
iss = delta*kss
Vss = np.log(kss**alpha-iss)

# capital grid
Nmid = (N-1)/2
gmin = -1.0
gmax = 1.0
lgrid = np.linspace(gmin,gmax,N+1)
lvals = 0.5*(lgrid[:-1]+lgrid[1:])
kvals = kss*np.exp(lvals)

# variables
V = tf.Variable(Vss*np.ones(N))
i = tf.Variable(np.zeros(N))

# derivatives
V_k = (V[1:]-V[:N-1])/(kvals[1:]-kvals[:N-1])

# equations
eq_V = beta*V[1:] - (tf.log(kvals[1:]**alpha-i[1:]) + (i[1:]-delta*kvals[1:])*V_k)
eq_k = 1 - (kvals[1:]**alpha-i[1:])*V_k
eq_bc1 = V[Nmid] - Vss
eq_bc2 = i[Nmid]
eq = tf.concat(0,[eq_V,eq_k,eq_bc1,eq_bc2])

# solve
