Bayesian Models
===============

Bernoulli distribution
----------------------

The Bernoulli distribution is a discrete distribution with boolean-valued outcome; 1 indicating *success* with probability :math:`p` and 0 indicating *failure*
with probability :math:`q = 1 -p`, where :math:`p \in [0, 1]`. The probability
mass function for :math:`k \in \{0, 1\}` is

.. math::

	f(k; p) = p^k (1-p)^{k-1} = \begin{cases} 1-p & \text{if } k = 0\\
	p & \text{if }k = 1, \end{cases}

and the cumulative distribution function is

.. math::

	F(k; p) = \begin{cases} 1-p & \text{if } k = 0\\
	1 & \text{if }k = 1. \end{cases}

The expected value and variance are as follows

.. math::

	\mathrm{E}[X] = p, \quad \mathrm{Var}[X]= p(1-p).	

The Bernoulli distribution is suitable for binary-outcome tests, for example, conversion rate (CRO) or click-through rate (CTR) testing.

.. autoclass:: cprior.BernoulliModel
	:members:
	:inherited-members:
	:show-inheritance:

.. autoclass:: cprior.BernoulliABTest
	:members:
	:inherited-members:	
	:show-inheritance:


Poisson distribution
--------------------

The Poisson distribution is a discrete distribution used to model occurrences and counts of rare events in an interval of time and/or space, when these are independent with constant rate :math:`\lambda`. The  probability mass function for :math:`k \in \mathbb{N}_0` is

.. math::

	f(k, \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

and the cumulative distribution function is

.. math::

	F(k; \lambda) = Q(\lfloor k + 1 \rfloor, \lambda),

where :math:`Q(a, z)` is the regularized incomplete gamma function and
:math:`\lfloor x \rfloor` is the floor function. Finally, the expected value and variance is :math:`\lambda`.

The Poisson distribution is applied to forecast arrival of customers for service at the checkout, visits to a website, etc.

.. autoclass:: cprior.PoissonModel
	:members:
	:inherited-members:
	:show-inheritance:
