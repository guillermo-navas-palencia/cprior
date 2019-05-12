Bernoulli-beta conjugate model
==============================

Posterior predictive distribution
---------------------------------

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

Proofs
------

Posterior predictive probability density function

.. math::

   f(x=0) &= \int_0^1 (1-p) \frac{p^{\alpha - 1} (1-p)^{\beta - 1}}{B(\alpha, \beta)} \mathop{dp}
   = \mathrm{E}[1-p] = \frac{\beta}{\alpha + \beta}.

   f(x=1) &= \int_0^1 p \frac{p^{\alpha - 1} (1-p)^{\beta - 1}}{B(\alpha, \beta)} \mathop{dp}
   = \mathrm{E}[p] = \frac{\alpha}{\alpha + \beta}.

Posterior predictive expected value

.. math::

   \mathrm{E}[X] = \mathrm{E}[\mathrm{E}[X | p]] = \mathrm{E}[p] = \frac{\alpha}{\alpha + \beta}.

Posterior predictive variance

.. math::

   \mathrm{Var}[X] = \mathrm{E}[X^2] - \mathrm{E}[X]^2 = \frac{\alpha}{\alpha + \beta} - \left(\frac{\alpha}{\alpha + \beta}\right)^2 = \frac{\alpha \beta}{(\alpha + \beta)^2}.
