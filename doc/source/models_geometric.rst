Geometric distribution
======================

The geometric distribution is a discrete probability distribution with parameter
:math:`p \in (0, 1)`. It can be defined as the number of Bernoulli trials, with
probability of success :math:`p`, required to obtain a success. The probability
mass function for :math:`k \ge 1` is

.. math::

   f(k; p) = (1 - p)^{k - 1} p,

and the cumulative distribution function is

.. math::

   F(k; p) = 1 - (1 - p)^k.

The expected value and variance are as follows

.. math::

   \mathrm{E}[X] = \frac{1}{p}, \quad \mathrm{Var}[X] = \frac{1 - p}{p^2}.

The geometric distribution is suitable to model the number of failures before
the first success.

.. autoclass:: cprior.GeometricModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.GeometricABTest
   :members:
   :inherited-members:
   :show-inheritance: