Negative binomial distribution
==============================

The negative binomial is a discrete probability distribution of the number of
successes in a sequence of Bernoulli independent trials with probability of
success :math:`p` before a number of :math:`r` failures occurs. The probability
mass function for :math:`k \in \{0, 1, \ldots, m\}` is

.. math::

   f(k; r, p) = \binom{k + r - 1}{r - 1}p^r (1-p)^k,

and the cumulative distribution function is

.. math::

   F(k; r, p) = I_{p}(r, 1 + k),

where :math:`I_x(a, b)` is the regularized incomplete beta function. The
expected value and variance are as follows

.. math::

   \mathrm{E}[X] = r\frac{1-p}{p}, \quad \mathrm{Var}[X] = r\frac{1-p}{p^2}.

The negative binomial distribution can be used to model how many clicks are
required before clicking a particular bottom of interest or stop clicking.

.. autoclass:: cprior.NegativeBinomialModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.NegativeBinomialABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.NegativeBinomialMVTest
   :members:
   :inherited-members:
   :show-inheritance: