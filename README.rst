CPrior
======

CPrior has functionalities to perform Bayesian statistics and running A/B and multivariate testing.

Example
-------

.. code::

   import scipy.stats as st

   from cprior.bernoulli import BernoulliModel
   from cprior.bernoulli import BernoulliABTest

   # Two model variants A and B, and build A/B test

   modelA = BernoulliModel()
   modelB = BernoulliModel()

   test = BernoulliABTest(modelA=modelA, modelB=modelB, simulations=1000000)