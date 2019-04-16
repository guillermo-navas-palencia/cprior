CPrior
======

CPrior has functionalities to perform Bayesian statistics and running A/B and multivariate testing.

Example
-------

.. code-block:: python

   import scipy.stats as st

   from cprior.bernoulli import BernoulliModel
   from cprior.bernoulli import BernoulliABTest

   # Two model variants A and B, and build A/B test

   modelA = BernoulliModel()
   modelB = BernoulliModel()

   test = BernoulliABTest(modelA=modelA, modelB=modelB, simulations=1000000)

   # Generate new data and update models

   data_A = st.bernoulli(p=0.10).rvs(size=1500, random_state=42)
   data_B = st.bernoulli(p=0.11).rvs(size=1600, random_state=42)

   test.update_A(data_A)
   test.update_B(data_B)
