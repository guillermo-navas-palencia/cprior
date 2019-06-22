|Travis|_ |Codecov|_

.. |Travis| image:: https://travis-ci.com/guillermo-navas-palencia/cprior.svg?branch=master
.. _Travis: https://travis-ci.com/guillermo-navas-palencia/cprior
    
.. |Codecov| image:: https://codecov.io/gh/guillermo-navas-palencia/cprior/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/guillermo-navas-palencia/cprior

CPrior
======

CPrior has functionalities to perform Bayesian statistics and running A/B and multivariate testing.

**Website**: http://gnpalencia.org/cprior/


Installation
------------

See http://gnpalencia.org/cprior/getting_started.html.

Dependencies
""""""""""""

CPrior requires:

* mpmath 1.0.0 or later. Website: http://mpmath.org/
* numpy 1.15.0 or later. Website: https://www.numpy.org/
* scipy 1.0.0 or later. Website: https://scipy.org/scipylib/
* pytest
* coverage

Testing
"""""""
Run all unit tests

.. code-block:: bash

   python setup.py test


Example
-------

A Bayesian A/B test with data following a Bernoulli distribution with two
distinct success probability. This example is a simple use case for
CRO (conversion rate) or CTR (click-through rate) testing.

.. code-block:: python

   import scipy.stats as st

   from cprior import BernoulliModel
   from cprior import BernoulliABTest

   modelA = BernoulliModel()
   modelB = BernoulliModel()

   test = BernoulliABTest(modelA=modelA, modelB=modelB, simulations=1000000)

   data_A = st.bernoulli(p=0.10).rvs(size=1500, random_state=42)
   data_B = st.bernoulli(p=0.11).rvs(size=1600, random_state=42)

   test.update_A(data_A)
   test.update_B(data_B)

   # Compute P[A > B] and P[B > A]
   print("P[A > B] = {:.4f}".format(test.probability(variant="A")))
   print("P[B > A] = {:.4f}".format(test.probability(variant="B")))

   # Compute posterior expected loss given a variant
   print("E[max(B - A, 0)] = {:.4f}".format(test.expected_loss(variant="A")))
   print("E[max(A - B, 0)] = {:.4f}".format(test.expected_loss(variant="B")))

The output should be the following:

.. code-block:: python

   P[A > B] = 0.1024
   P[B > A] = 0.8976
   E[max(B - A, 0)] = 0.0147
   E[max(A - B, 0)] = 0.0005
