Poisson distribution
====================

The Poisson distribution is a discrete distribution used to model occurrences and counts of rare events in an interval of time and/or space, when these are independent with constant average event rate :math:`\lambda`. The probability mass function for :math:`k \in \mathbb{N}_0` is

.. math::

   f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

and the cumulative distribution function is

.. math::

   F(k; \lambda) = Q(1 + \lfloor k \rfloor, \lambda),

where :math:`Q(a, z)` is the regularized incomplete gamma function and
:math:`\lfloor x \rfloor` is the floor function. Finally, the expected value and variance is :math:`\lambda`.

The Poisson distribution is applied to forecast arrival of customers for service at the checkout or visits to a website.

.. autoclass:: cprior.PoissonModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.PoissonABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.PoissonMVTest
   :members:
   :inherited-members:
   :show-inheritance:
