Uniform-Pareto conjugate model
==============================

Posterior predictive distribution
---------------------------------

If :math:`X|\theta \sim \mathcal{U}(0, \theta)` with :math:`\theta \sim
\mathcal{PA}(\alpha, \beta)`, then the posterior predictive probability
density function, the expected value and variance of :math:`X` are

.. math::

   f(x; \alpha, \beta) = \begin{cases}
      \frac{\alpha}{(\alpha + 1) \beta}, & 0 < x < \beta,\\
      \frac{\alpha \beta^{\alpha}}{(\alpha + 1)x^{\alpha + 1}}, & x \ge \beta
   \end{cases}

.. math::

   \mathrm{E}[X] = \frac{\alpha \beta}{2(\alpha - 1)}, \quad \mathrm{Var}[X] = \frac{\alpha (\alpha^2 - 2\alpha + 4)\beta^2}{12(\alpha - 1)^2 (\alpha - 2)},

where :math:`\mathrm{E}[X]` is defined for :math:`\alpha > 1` and
:math:`\mathrm{Var}[X]` is defined for :math:`\alpha > 2`.

Proofs
------

Posterior predictive probability density function

.. math::

   f(x; \alpha, \beta) = \int_x^{\infty} \frac{1}{\theta} \frac{\alpha \beta^{\alpha}}{\theta^{\alpha + 1}} \mathop{d\theta}.

if :math:`x \in (0, \beta)` then

.. math::

   f(x; \alpha, \beta) = \int_0^{\beta} \frac{1}{\theta} \frac{\alpha \beta^{\alpha}}{\theta^{\alpha + 1}} \mathop{d\theta} = \frac{\alpha}{(\alpha + 1) \beta},

otherwise, if :math:`x \ge \beta`,

.. math::

   f(x; \alpha, \beta) = \int_x^{\infty} \frac{1}{\theta} \frac{\alpha \beta^{\alpha}}{\theta^{\alpha + 1}} \mathop{d\theta} = \frac{\alpha \beta^{\alpha}}{(\alpha + 1)x^{\alpha + 1}}.


Posterior predictive expected value

.. math::

   \mathrm{E}[X] = \mathrm{E}[\mathrm{E}[X | \theta]] = \mathrm{E}\left[\frac{\theta}{2}\right] = \frac{\alpha \beta}{2(\alpha - 1)}.


Posterior predictive variance

.. math::

   \mathrm{Var}[X] &= \mathrm{E}[\mathrm{Var}[X | \theta]] + \mathrm{Var}[\mathrm{E}[X | \theta]]\\
   &= \mathrm{E}\left[\frac{\theta^2}{12}\right] + \mathrm{V}\left[\frac{\theta}{2}\right] = \frac{1}{12}\mathrm{E}[\theta^2] + \frac{1}{4}\mathrm{V}[\theta]\\
   &= \frac{\alpha \beta^2}{12(\alpha - 2)} + \frac{\alpha \beta^2}{4(\alpha - 1)^2(\alpha - 2)} = \frac{\alpha (\alpha^2 - 2\alpha + 4)\beta^2}{12(\alpha - 1)^2 (\alpha - 2)},

the same result if obtained using

.. math::

   \mathrm{Var}[X] &= \mathrm{E}[X^2] - \mathrm{E}[X]^2 = \mathrm{E}\left[\frac{\theta^2}{3}\right] - \left(\frac{\alpha \beta}{2(\alpha - 1)}\right)^2\\
   &= \frac{\alpha \beta^2}{3(\alpha - 2)} - \frac{\alpha^2 \beta^2}{4(\alpha - 1)^2} = \frac{\alpha (\alpha^2 - 2\alpha + 4)\beta^2}{12(\alpha - 1)^2 (\alpha - 2)}.