Negative binomial-beta conjugate model
======================================

Posterior predictive distribution
---------------------------------

If :math:`X|p \sim \mathcal{NB}(r, p)` with :math:`p \sim \mathcal{B}(\alpha,
\beta)`, then the posterior predictive probability density function, the
expected value and variance of :math:`X` are

.. math::

   f(x; r, \alpha, \beta) = \binom{x + r - 1}{r - 1}\frac{B(\alpha + r,
   \beta + x)}{B(\alpha, \beta)}, \quad x = 0, 1, 2, \ldots.

.. math::

   \mathrm{E}[X] = r\frac{\beta}{\alpha - 1}, \quad \mathrm{Var}[X] =
   \frac{r \beta (\alpha + r - 1)(\alpha + \beta - 1)}{(\alpha - 1)^2
   (\alpha - 2)},

where :math:`\mathrm{E}[X]` is defined for :math:`\alpha > 1` and
:math:`\mathrm{Var}[X]` is defined for :math:`\alpha > 2`.

Proofs
------

Posterior predictive probability density function

.. math::

   f(x; r, \alpha, \beta) &= \int_0^1 \binom{x + r - 1}{r - 1}p^r (1-p)^x
   \frac{p^{\alpha - 1} (1-p)^{\beta - 1}}{B(\alpha, \beta)} \mathop{dp}\\
   &= \binom{x + r - 1}{r - 1}\frac{1}{B(\alpha, \beta)}
   \int_0^1 p^{\alpha + r - 1} (1-p)^{\beta + x - 1} \mathop{dt}
   = \binom{x + r - 1}{r - 1}\frac{B(\alpha + r,
   \beta + x)}{B(\alpha, \beta)}.


Posterior predictive expected value

.. math::

   \mathrm{E}[X] = \mathrm{E}[\mathrm{E}[X | p]] = r \mathrm{E}
   \left[\frac{1 - p}{p}\right] = r\frac{\beta}{\alpha - 1}.


Posterior predictive variance

.. math::

   \mathrm{Var}[X] &= \mathrm{E}[\mathrm{Var}[X | p]] + \mathrm{Var}[\mathrm{E}[X | p]]\\
   &= \mathrm{E}\left[r\frac{1 - p}{p^2}\right] + \mathrm{Var}\left[r\frac{1-p}{p}\right]
   = r \left(\mathrm{E}\left[\frac{1}{p^2}\right] - \mathrm{E}\left[\frac{1}{p}\right]\right) + r^2 \mathrm{Var}\left[\frac{1}{p}\right]\\
   &= r \frac{\alpha + \beta - 1}{\alpha - 1}\left(\frac{\alpha + \beta - 2}{\alpha - 2} - 1\right) + r^2\frac{\beta^2 (\alpha + \beta - 1)}{(\alpha - 1)^2(\alpha - 2)} = \frac{r \beta (\alpha + r - 1)(\alpha + \beta - 1)}{(\alpha - 1)^2
   (\alpha - 2)}.

.. note::

   Recall the calculation of :math:`\mathrm{E}\left[\frac{1}{p}\right]`,
   :math:`\mathrm{E}\left[\frac{1}{p^2}\right]` and :math:`\mathrm{Var}\left[\frac{1}{p}\right]` derived for the geometric-beta model.
