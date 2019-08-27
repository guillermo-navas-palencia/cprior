Binomial distribution
=====================

The binomial distribution is the discrete probability distribution of the number
of successes in a sequence of :math:`m` boolean-valued outcome independent
trials with probability of success :math:`p`. The probability mass function
for :math:`k \in \{0, 1, \ldots, m\}` is

.. math::

   f(k; m, p) = \binom{m}{k} p^k (1-p)^{m-k},

and the cumulative distribution function is

.. math::

   F(k; m, p) = I_{1-p}(m - k, 1 + k),

where :math:`I_x(a, b)` is the regularized incomplete beta function. The
expected value and variance are as follows

.. math::

   \mathrm{E}[X] = mp, \quad \mathrm{Var}[X] = mp(1-p).

The Bernoulli distribution is suitable for binary-outcome tests, for example,
CRO (conversion rate) or CTR (click-through rate) testing.

.. autoclass:: cprior.models.BinomialModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.models.BinomialABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.models.BinomialMVTest
   :members:
   :inherited-members:
   :show-inheritance: