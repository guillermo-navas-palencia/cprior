Normal-inverse-gamma distribution
=================================

The probability density function of the normal-inverse gamma distribution
:math:`\mathcal{N}\Gamma^{-1}(\mu, \lambda, \alpha, \beta)` with location
parameter :math:`\mu`, variance scale parameter :math:`\lambda > 0`, shape
parameter :math:`\alpha > 0` and scale parameter :math:`\beta > 0,` for
:math:`x \in \mathbb{R}` and :math:`\sigma^2 \in \mathbb{R}^+`, is given by

.. math::

   f(x,\sigma^2; \mu,\lambda,\alpha,\beta) =  \frac {\sqrt{\lambda}} {\sigma\sqrt{2\pi} } \, \frac{\beta^\alpha}{\Gamma(\alpha)} \, \left( \frac{1}{\sigma^2} \right)^{\alpha + 1}   \exp \left( -\frac { 2\beta + \lambda(x - \mu)^2} {2\sigma^2}  \right),

and the cumulative distribution function is

.. math::

   F(x,\sigma^2; \mu,\lambda,\alpha,\beta) =  \frac{e^{-\frac{\beta}{\sigma^2}} \left(\frac{\beta }{\sigma ^2}\right)^\alpha
      \left(\operatorname{erf}\left(\frac{\sqrt{\lambda} (x-\mu )}{\sqrt{2} \sigma }\right)+1\right)}{2
      \sigma^2 \Gamma (\alpha)}.

The expected value and variance are as follows

   .. math::

      \mathrm{E}[x] &= \mu, \quad \mathrm{E}[\sigma^2] = \frac{\beta}{\alpha-1}, \; \alpha > 1.

      \mathrm{Var}[x] &= \frac{\beta}{(\alpha - 1)\lambda}, \; \alpha > 1,
      \quad \mathrm{Var}[\sigma^2] = \frac{\beta^2}{(\alpha-1)^2(\alpha - 2)}, \; \alpha > 2.

The normal-inverse-gamma distribution is used as a conjugate prior distribution for
the normal distribution with unknown mean and variance.


.. autoclass:: cprior.cdist.NormalInverseGammaModel
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.cdist.NormalInverseGammaABTest
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: cprior.cdist.NormalInverseGammaMVTest
   :members:
   :inherited-members:
   :show-inheritance: