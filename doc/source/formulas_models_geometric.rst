Geometric-beta conjugate model
==============================

Posterior predictive distribution
---------------------------------

If :math:`X|p \sim \mathcal{G}(p)` with :math:`p \sim \mathcal{B}(\alpha, \beta)`,
then the posterior predictive probability density function, the expected value
and variance of :math:`X` are


.. math::

   f(x; \alpha, \beta) = \frac{B(\alpha + 1, \beta + x - 1)}{B(\alpha, \beta)},
   \quad x = 0, 1, 2, \ldots.

.. math::

   \mathrm{E}[X] = \frac{\alpha + \beta - 1}{\alpha - 1}, \quad \mathrm{Var}[X] =
   \frac{\beta (\alpha + \beta - 1)}{(\alpha - 1)^2 (\alpha - 2)},

where :math:`\mathrm{E}[X]` is defined for :math:`\alpha > 1` and
:math:`\mathrm{Var}[X]` is defined for :math:`\alpha > 2`.

Proofs
------

Posterior predictive probability density function

.. math::

   f(x; \alpha, \beta) &= \int_0^1 (1 - p)^{x - 1} p \frac{p^{\alpha - 1} (1-p)^{\beta - 1}}{B(\alpha, \beta)} \mathop{dp}\\
   &= \frac{1}{B(\alpha, \beta)} \int_0^1 p^{\alpha} (1-p)^{\beta + x - 2} \mathop{dp}
   = \frac{B(\alpha + 1, \beta + x - 1)}{B(\alpha, \beta)}.


Posterior predictive expected value

.. math::

   \mathrm{E}[X] = \mathrm{E}[\mathrm{E}[X | p]] = \mathrm{E}\left[\frac{1}{p}\right] = \frac{\alpha + \beta - 1}{\alpha - 1}.

Note that,

.. math::

   \mathrm{E}\left[\frac{1}{p}\right] = \int_0^1 \frac{1}{p} \frac{p^{\alpha - 1} (1-p)^{\beta - 1}}{B(\alpha, \beta)} \mathop{dp} = \int_0^1 \frac{p^{\alpha - 2} (1-p)^{\beta - 1}}{B(\alpha - 1, \beta)} \frac{\alpha + \beta - 1}{\alpha - 1} \mathop{dp} = \frac{\alpha + \beta - 1}{\alpha - 1},

where we use the property of the beta function: :math:`B(a -1, b) = \frac{a + b - 1}{a - 1} B(a, b)`.


Posterior predictive variance

.. math::

   \mathrm{Var}[X] = \mathrm{E}[X^2] - \mathrm{E}[X]^2 = \frac{\beta (\alpha + \beta - 1)}{(\alpha - 1)^2 (\alpha - 2)}.

Similarly, we have that

.. math::

   \mathrm{E}[X^2] = \frac{\alpha + \beta - 1}{\alpha - 1}\frac{\alpha + \beta - 2}{\alpha - 2}

and

.. math::

   \mathrm{Var}[X] = \frac{\alpha + \beta - 1}{\alpha - 1}\frac{\alpha + \beta - 2}{\alpha - 2} - \left(\frac{\alpha + \beta - 1}{\alpha - 1}\right)^2 = \frac{\beta (\alpha + \beta - 1)}{(\alpha - 1)^2 (\alpha - 2)}.


.. note::

   The same can be proven applying properties of the beta and beta prime distribution. Given that if :math:`X \sim \mathcal{B}(a, b) \rightarrow \frac{X}{1 - X} \sim \beta'(a, b)` and if :math:`Y \sim \beta'(a, b) \rightarrow \frac{1}{Y} \sim \beta'(b, a)`, we get that if :math:`X \sim \mathcal{B}(a, b)` then :math:`\frac{1 - X}{X} \sim \beta'(b, a)`, thus

   .. math::

      \mathrm{E}\left[\frac{1}{X}\right] = \frac{\beta}{\alpha - 1} + 1 = \frac{\alpha + \beta - 1}{\alpha - 1},

   and

   .. math::

      \mathrm{Var}\left[\frac{1}{X}\right] = \frac{\beta (\alpha + \beta - 1)}{(\alpha - 1)^2 (\alpha - 2)}.
