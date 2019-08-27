Uniform distribution
====================

The uniform distribution is a continuous distribution with constant probability in its support defined by the two parameters, :math:`a` and :math:`b`, which are its minimum and maximum values. The probability density function for :math:`x \in [a, b]` is given by

.. math::

   f(x; a, b) = \frac{1}{b-a},

and :math:`0` elsewhere. The cumulative distribution is

.. math::

   F(x; a, b) = \begin{cases}
      0, & x < a\\
      \frac{x-a}{b-a}, & x \in [a, b)\\
      1, & x \ge b
   \end{cases}

The expected value and variance are as follows

.. math::

   \mathrm{E}[X] = \frac{a + b}{2}, \quad \mathrm{Var}[X] = \frac{(b-a)^2}{12}.

The uniform distribution is used to model events that are equally likely.

.. autoclass:: cprior.models.UniformModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.models.UniformABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.models.UniformMVTest
   :members:
   :inherited-members:
   :show-inheritance:
