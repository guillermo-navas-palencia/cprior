Poisson-gamma conjugate model
=============================

Posterior predictive distribution
---------------------------------

If :math:`X|\lambda \sim \mathcal{P}(\lambda)` with :math:`\lambda \sim \mathcal{G}(\alpha, \beta)`, then the posterior predictive probability density function, the expected value and variance of :math:`X` are

.. math::

   f(x; \alpha, \beta) = \binom{x + \alpha -1}{\alpha - 1}\left(\frac{\beta}{\beta+1}\right)^{\alpha}\left(\frac{1}{\beta+1}\right)^x, \quad x = 0, 1, 2, \ldots.

.. math::

   \mathrm{E}[X] = \frac{\alpha}{\beta}, \quad \mathrm{Var}[X] = \frac{\alpha (\beta + 1)}{\beta^2}.

Proofs
------

Posterior predictive probability density function

.. math::

   f(x; \alpha, \beta) &= \int_0^{\infty} \frac{\lambda^x e^{-\lambda}}{x!} \frac{\beta^{\alpha} \lambda^{\alpha - 1} e^{-\beta \lambda}}{\Gamma(\alpha)} \mathop{d\lambda}\\
   &= \frac{\beta^{\alpha}}{\Gamma(a) x!} \int_0^{\infty} \lambda^{\alpha + x - 1} e^{-\lambda(\beta + 1)} \mathop{d\lambda} = \frac{\beta^{\alpha}}{\Gamma(a) x!} \frac{\Gamma(\alpha + x)}{(\beta + 1)^{\alpha + x}}\\
   &= \binom{x + \alpha -1}{\alpha - 1}\left(\frac{\beta}{\beta+1}\right)^{\alpha}\left(\frac{1}{\beta+1}\right)^x.


Note that this is the probability density function of the negative binomial distribution, thus

.. math::

   X \sim \mathcal{NB}\left(\alpha, \frac{\beta}{\beta + 1}\right),

see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html. Note that this definition slightly differs from https://en.wikipedia.org/wiki/Negative_binomial_distribution.

Posterior predictive expected value

.. math::

   \mathrm{E}[X] = \mathrm{E}[\mathrm{E}[X | \lambda]] = \mathrm{E}[\lambda] = \frac{\alpha}{\beta}.

Posterior predictive variance

Applying http://mathworld.wolfram.com/NegativeBinomialDistribution.html Eq. (24) with :math:`r = \alpha`, :math:`p = \frac{\beta}{\beta + 1}` and :math:`q = 1 - p` we get

.. math::

   \mathrm{Var}[X] = \frac{\alpha / (\beta + 1)}{\beta^2 / (\beta + 1)^2} = \frac{\alpha (\beta + 1)}{\beta^2}.