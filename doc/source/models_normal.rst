Normal distribution
===================

The normal distribution or Gaussian distribution is a continuous probability
distribution. The probability density function of a normal distribution with
mean :math:`\mu` and standard deviation :math:`\sigma` for
:math:`x \in \mathbb{R}` is

.. math::

   f(x; \mu, \sigma) = \frac{\exp\left(-\frac{1}{2}
   \left(\frac{x-\mu}{\sigma}\right)^2\right)}{\sigma\sqrt{2 \pi}},

and the cumulative distribution is

.. math::

   F(x; \mu, \sigma) = \frac{1}{2}\left(1 + \mathrm{erf}\left(\frac{x-\mu}
   {\sigma\sqrt{2}}\right)\right).

The expected value and variance are as follows

.. math::

   \mathrm{E}[X] = \mu. \quad \mathrm{Var}[X] = \sigma^2.

The normal distribution is used to model/approximate symmetric centralized
distributions.


.. autoclass:: cprior.models.NormalModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.models.NormalABTest
   :members:
   :inherited-members:  
   :show-inheritance:

.. autoclass:: cprior.models.NormalMVTest
   :members:
   :inherited-members:
   :show-inheritance: