Normal-normal-inverse-gamma conjugate model
===========================================

Posterior predictive distribution
---------------------------------

If :math:`X| \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)` with
:math:`(\mu, \sigma) \sim \mathcal{N}\Gamma^{-1}(\mu_0, \lambda, \alpha, \beta)`,
then the posterior predictive probability density function, the expected
value and variance of :math:`X` are

.. math::

   f(x; \mu_0, \lambda, \alpha, \beta) = \frac{\alpha}{\beta(1 + \lambda^{-1})}
   \frac{\left(1 + \frac{1}{2\alpha} \left(\frac{\alpha(x - \mu_0)}{\beta(1+\lambda^{-1})} \right)^2 \right)^{-\alpha - 1/2}}
   {\sqrt{2\alpha}B(\alpha, 1/2)},

.. math::

   \mathrm{E}[X] = \mu_0, \quad \mathrm{Var}[X] = \frac{\left(\beta(1 +
   \lambda^{-1})\right)^2}{\alpha(\alpha - 1)},

where :math:`\mathrm{E}[X]` is defined for :math:`\alpha > 1/2` and
:math:`\mathrm{Var}[X]` is defined for :math:`\alpha > 1`.


Proofs
------

Posterior predictive probability density function

Note that this is the probability density function of the
Student's t-distribution, thus

.. math::

   X \sim t_{2 \alpha}\left(\mu_0, \frac{\beta (1 + \lambda^{-1})}{\alpha}\right),

see :cite:`Murphy2007`.


Posterior predictive expected value

Apply properties of the Student's t-distribution.

.. math ::

   \mathrm{E}[X] = \mu_0.


Posterior predictive variance

Apply properties of the Student's t-distribution.

.. math::

   \mathrm{Var}[X] = \frac{\left(\beta(1 +
   \lambda^{-1})\right)^2}{\alpha(\alpha - 1)}.


References
----------

.. bibliography:: refs.bib
   :filter: docname in docnames