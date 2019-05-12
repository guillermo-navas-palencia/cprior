Exponential-gamma conjugate model
=================================

Posterior predictive distribution
---------------------------------

If :math:`X|\lambda \sim \mathcal{E}(\lambda)` with :math:`\lambda \sim \mathcal{G}(\alpha, \beta)`, then the posterior predictive probability density function, the expected value and variance of :math:`X` are

.. math::

   f(x; \alpha, \beta) = \frac{\alpha \beta^{\alpha}}{(\beta + x)^{\alpha + 1}}, \quad x \ge 0.

.. math::

   \mathrm{E}[X] = \frac{\beta}{\alpha - 1}, \quad \mathrm{Var}[X] = \frac{\alpha \beta^2}{(\alpha - 1)^2 (\alpha - 2)},

where :math:`\mathrm{E}[X]` is defined for :math:`\alpha > 1` and
:math:`\mathrm{Var}[X]` is defined for :math:`\alpha > 2`.

Proofs
------

Posterior predictive probability density function


.. math::

   f(x; \alpha, \beta) &= \int_0^{\infty} \lambda e^{-\lambda x} \frac{\beta^{\alpha} \lambda^{\alpha - 1} e^{-\beta \lambda}}{\Gamma(\alpha)} \mathop{d\lambda} = \frac{\beta^{\alpha}}{\Gamma(a)} \int_0^{\infty} \lambda^{\alpha} e^{-\lambda(\beta + x)} \mathop{d\lambda}\\
   &= \frac{\beta^{\alpha}}{\Gamma(a)}\frac{\Gamma(a + 1)}{(\beta + x)^{\alpha + 1}} = \frac{\alpha \beta^{\alpha}}{(\beta + x)^{\alpha + 1}}.


Note that this is the probability density function of the Lomax distribution, thus

.. math::

   X \sim \mathcal{Lomax}(\alpha, \beta),

see https://en.wikipedia.org/wiki/Lomax_distribution.


Posterior predictive expected value

.. math::

   \mathrm{E}[X] =  \mathrm{E}[\mathrm{E}[X | \lambda]] = \mathrm{E}\left[\frac{1}{\lambda}\right],

The reciprocal of the gamma distribution follows an inverse gamma distribution with expected value :math:`\frac{\beta}{\alpha - 1}`.


Posterior predictive variance

Instead of directly applying well-known properties of the Lomax distribution, we use the law of total variance,

.. math::

   \mathrm{Var}[X] &= \mathrm{E}[\mathrm{Var}[X | \lambda]] + \mathrm{Var}[\mathrm{E}[X | \lambda]] = E\left[\frac{1}{\lambda^2}\right] + \mathrm{Var}\left[\frac{1}{\lambda}\right]\\
   &= \frac{\beta^2}{(\alpha - 1)(\alpha - 2)} + \frac{\beta^2}{(\alpha - 1)^2(\alpha - 2)} = \frac{\alpha \beta^2}{(\alpha - 1)^2 (\alpha - 2)},

where we use the second moment and the variance of the inverse gamma distribution.
