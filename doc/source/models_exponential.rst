Exponential distribution
========================

THe exponential distribution is the probability distribution that describes
the waiting times between successive events following a Poisson distribution
with constant average rate :math:`\lambda`. The probability density function
for :math:`x \ge 0` is given by

.. math::

   f(x; \lambda) = \lambda e^{-\lambda x},

and the cumulative distribution function is

.. math::

   F(x; \lambda) = 1 - e^{-\lambda x}.

The expected value and variance are as follows

.. math::

   \mathrm{E}[X] = \frac{1}{\lambda}. \quad \mathrm{Var}[X] = \frac{1}{\lambda^2}.


The exponential distribution is often used to test revenue metrics like ARPPU
(average revenue per paying user), i.e, ARPPU = Revenue / Paying Users.


.. autoclass:: cprior.ExponentialModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.ExponentialABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.ExponentialMVTest
   :members:
   :inherited-members:
   :show-inheritance:
