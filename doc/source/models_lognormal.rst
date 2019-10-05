Log-normal distribution
=======================

The log-normal distribution is a continuous probability distribution of a
random variable of which logarithm is normally distributed. The probability
density function of a log-normal distribution with mean :math:`\mu` and
standard deviation :math:`\sigma` for :math:`x > 0` is

.. math::

   f(x; \mu, \sigma) = \frac{\exp\left(-\frac{1}{2}
   \left(\frac{\log(x)-\mu}{\sigma}\right)^2\right)}{x\sigma\sqrt{2 \pi}},

and the cumulative distribution is

.. math::

   F(x; \mu, \sigma) = \frac{1}{2}\left(1 + \mathrm{erf}\left(\frac{\log(x)-\mu}
   {\sigma\sqrt{2}}\right)\right).

The expected value and variance are as follows

.. math::

   \mathrm{E}[X] = \exp\left(\mu + \frac{\sigma^2}{2}\right). \quad
   \mathrm{Var}[X] = \left(\exp(\sigma^2) - 1\right) \exp(2\mu + \sigma^2).

The log-normal distribution is often used to test revenue metrics (see also the
exponential distribution) or time spent on a web page.


.. autoclass:: cprior.models.LogNormalModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.models.LogNormalABTest
   :members:
   :inherited-members:  
   :show-inheritance:

.. autoclass:: cprior.models.LogNormalMVTest
   :members:
   :inherited-members:
   :show-inheritance: