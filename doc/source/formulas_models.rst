Formulas for Bayesian models
============================

Bernoulli distribution
----------------------

If :math:`X|p \sim \mathcal{BE}(p)` with :math:`p \sim \mathcal{B}(\alpha, \beta)`,
then the posterior predictive probability density function, the expected value
and variance of :math:`X` are

.. math::

   f(x; \alpha, \beta) = \begin{cases}
      \frac{\beta}{\alpha + \beta} & \text{if $x = 0$}\\
      \frac{\alpha}{\alpha + \beta} & \text{if $x = 1$},
      \end{cases}

.. math::

   \mathrm{E}[X] =  \frac{\alpha}{\alpha + \beta}, \quad \mathrm{Var}[X] = \frac{\alpha \beta}{(\alpha + \beta)^2}.

Poisson distribution
--------------------

If :math:`X|\lambda \sim \mathcal{P}(\lambda)` with :math:`\lambda \sim \mathcal{G}(\alpha, \beta)`, then the posterior predictive probability density function, the expected value and variance of :math:`X` are

.. math::

   f(x; \alpha, \beta) = \binom{x + \alpha -1}{\alpha - 1}\left(\frac{\beta}{\beta+1}\right)^{\alpha}\left(\frac{1}{\beta+1}\right)^x, \quad x = 0, 1, 2, \ldots.

.. math::

   \mathrm{E}[X] = \frac{\alpha}{\beta}, \quad \mathrm{Var}[X] = \frac{\alpha (\beta + 1)}{\beta^2}.
