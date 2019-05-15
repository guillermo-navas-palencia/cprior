Binomial-beta conjugate model
=============================

Posterior predictive distribution
---------------------------------

If :math:`X|p \sim \mathcal{BI}(m, p)` with :math:`p \sim \mathcal{B}(\alpha, \beta)`,
then the posterior predictive probability density function, the expected value
and variance of :math:`X` are

.. math::

   f(x; m, \alpha, \beta) = \binom{m}{x}\frac{B(\alpha + x, m - x + \beta)}
   {B(\alpha, \beta)}, \quad x = 0, 1, 2, \ldots.

.. math::

   \mathrm{E}[X] = m\frac{\alpha}{\alpha + \beta}, \quad \mathrm{Var}[X] = 
   \frac{m \alpha \beta (m + \alpha + \beta)}{(\alpha + \beta)^2
   (\alpha + \beta + 1)}.


Proofs
------

Posterior predictive probability density function

.. math::

   f(x; m, \alpha, \beta) &= \int_0^1 \binom{m}{x}p^x (1-p)^{m - x}
   \frac{p^{\alpha - 1} (1-p)^{\beta - 1}}{B(\alpha, \beta)} \mathop{dp}\\
   &= \binom{m}{x}\frac{1}{B(\alpha, \beta)} \int_0^1 p^{\alpha + x - 1}
   (1-p)^{\beta + m - x - 1} \mathop{dp} = \binom{m}{x}\frac{B(\alpha + x, m - x + \beta)}
   {B(\alpha, \beta)},

Note that this is the probability density function of the beta-binomial distribution, thus

.. math::

   X \sim \mathcal{BB}(m, \alpha, \beta),

see https://en.wikipedia.org/wiki/Beta-binomial_distribution.


Posterior predictive expected value

.. math::

   \mathrm{E}[X] = \mathrm{E}[\mathrm{E}[X | p]] = \mathrm{E}[mp] = m\frac{\alpha}{\alpha + \beta}.


Posterior predictive variance

Applying properties of the beta-binomial distribution, we obtain

.. math::

   \mathrm{Var}[X] = \frac{m \alpha \beta (m + \alpha + \beta)}{(\alpha + \beta)^2
   (\alpha + \beta + 1)}.