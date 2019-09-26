meteo
=====

Homotopy and constrained gradient descent with Python and Tensorflow.

Goals
-----

1. Traverse homotopy paths efficiently. In real space, this is basically solved. Moving to the complex domain is more difficult, as it would currently require the user to derive the relevant formulas in the complex domain by hand. This should be automated (call this 1a).
2. Solve constrained optimization problems efficiently. The relevant use case here would be something like maximum likelihood (or method of moments) wherein there are certain equilibrium constraints that must be satisfied.

It would seem that both can be accomplished in tensorflow. The approach here would be to have the user define their model in tensorflow, then provide new operations that act on these.

Todo
----

+ Cross-derivatives and discrete dimensions
+ Sparsity detection
+ Improve discretization, Runge-Kutta maybe?
+ Regularization and perturbation
