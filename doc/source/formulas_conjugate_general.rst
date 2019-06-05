General formulas and definitions
================================

Formulas for A/B testing
------------------------

The main metrics to perform A/B testing are described in :cite:`Stucchio2015`. Let us consider two variants :math:`X_A` and :math:`X_B` for testing.

The **error probability** or probability of :math:`X_B > X_A` is denoted as

.. math::
   P[X_B > X_A] = \int_{-\infty}^{\infty} \int_{x_A}^{\infty} f(x_A, x_B) \mathop{dx_B} \mathop{dx_A},

where :math:`f(x_A, x_B)` is the joint probability distribution, under the assumption of independence, i.e. :math:`f(x_A, x_B) = f(x_A) f(x_B)`.

The **expected loss function** given a joint posterior is the expected value of the **loss function**. The loss function is the expected uplift lost by choosing a given variant. If variant :math:`X_B` is chosen we have

.. math::

   EL(X_B) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max(x_A - x_B, 0) f(x_A, x_B) \mathop{dx_B} \mathop{dx_A}.

Other metrics also considered are the **relative expected loss** or uplift and credible intervals. A credible interval is a region which has a specified probability of containing the true value.

Formulas for Multivariate testing
---------------------------------

Let us first introduce some properties of the distribution of the maximum of a set of i.i.d. random variables with support on the whole real line.

.. math::

   X_{max} = \max\{X_1, \ldots, X_n\}


The cumulative distribution function is

.. math::

   F_{X_{max}}(z) = P\left[\underset{i=1, \ldots, n}\max{X_i} < z\right] = \prod_{i=1}^n P[X_i \le z] = \prod_{i=1}^n F_{X_i}(z),

where :math:`F_{X_i}(z)` is the cdf of each random variable :math:`X_i`. The probability density functions is obtain after derivation

.. math::

   f_{X_{max}}(z) = \sum_{i=1}^n f_{X_i}(z) \prod_{j \neq i} F_{X_j}(z).

where :math:`f_{X_i}(z)` is the pdf of each random variable :math:`X_i`.

The **probability to beat all** is defined as

.. math::

   P\left[X_i \ge \underset{j \neq i}\max{X_j}\right] = \int_{-\infty}^{\infty} f(x_i) \prod_{j \neq i} F_{X_j}(x_i) \mathop{dx_i}.


The **expected loss function vs all** is defined as

.. math::

   \mathrm{E}[\max(\underset{j \neq i}\max{X_j} - X_i, 0)]

Take :math:`Y = \underset{j \neq i}\max{X_j}`, then we have

.. math::

   EL(X_i) &= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max(y - x_i, 0) f(y) f(x_i) \mathop{dx_i} \mathop{dy} \\
   &= \int_{-\infty}^{\infty} \int_{-\infty}^y y f(y)f(x_i) \mathop{dx_i} \mathop{dy} - \int_{-\infty}^{\infty} \int_{-\infty}^y x_i f(y)f(x_i) \mathop{dx_i} \mathop{dy}\\
   &= \int_{-\infty}^{\infty} y f(y) F_{X_i}(y) \mathop{dy} - \int_{-\infty}^{\infty} f(y)
   F^*_{X_i}(y) \mathop{dy},

where :math:`F^*_{X_i}(y) = \int_{-\infty}^y x_i f(x_i) \mathop{dx_i}`.

References
----------

.. bibliography:: refs.bib
   :filter: docname in docnames
