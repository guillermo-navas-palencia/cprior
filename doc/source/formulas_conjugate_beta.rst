Beta distribution
=================

The error probability or chance to beat
---------------------------------------

Given two distributions :math:`X_A \sim \mathcal{B}(\alpha_A, \beta_A)` and :math:`X_B \sim \mathcal{B}(\alpha_B, \beta_B)` such that :math:`(\alpha_A, \beta_A, \alpha_B, \beta_B) \in \mathbb{R}_+^4`, :math:`P[X_B > X_A]` is given by

.. math::

   P[X_B > X_A] = 1 - \frac{B(\alpha_A + \alpha_B, \beta_A)}{\alpha_B B(\alpha_A, \beta_A)B(\alpha_B, \beta_B)} 
   \,_3F_2\left({\alpha_B, \alpha_A+\alpha_B, 1-\beta_B \atop 1+\alpha_B, \alpha_A + \alpha_B + \beta_A};1\right),
   
where :math:`B(a,b)` is the beta function and :math:`_3F_2(a,b,c;d,e;z)` is the generalized hypergeometric function. Note that the hypergeometric function is terminating for :math:`\beta_B \in \mathbb{N}`, although is numerically unstable due to the amount of cancellation when adding large alternating terms. A stable convergent solution is given by applying the transformation in http://functions.wolfram.com/07.27.17.0035.01 yielding

.. math::

   P[X_B > X_A] = 1-\frac{B(\alpha_A + \alpha_B, \beta_A + \beta_B)}{\beta_A B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)}\, 
   _3F_2\left({1, \alpha_A + \beta_A, \beta_A + \beta_B \atop \beta_A + 1, \alpha_A + \alpha_B + \beta_A + \beta_B};1\right)

The hypergeometric function is reduced to

.. math::

   \sum_{k=0}^{\infty} \frac{(\alpha_A + \beta_A)_k (\beta_A + \beta_B)_k}{(\beta_A + 1)_k (\alpha_A + \alpha_B + \beta_A + \beta_B)_k},

where all terms are positive. For some combinations of parameters, this hypergeometric series is slowly convergent, therefore series acceleration techniques come into play. Note that above expression can be written as a
terminating series in terms of the beta function when at least one of the
parameters is a positive integer number. See special cases in :cite:`Miller2015`.

.. note::

   The special cases in CPrior are efficiently implemented using a backward  recurrence, avoiding underflow and overflow issues and computation of negligible terms.

The expected loss function
--------------------------

The expected loss function can easily be calculated from the `definition <formulas_conjugate_general.html>`__ yielding

.. math::
   \mathrm{EL}(X_B) = \frac{\alpha_A}{\alpha_A + \beta_A} f(\alpha_B, \beta_B, \alpha_A + 1, \beta_A) - \frac{\alpha_B}{\alpha_B + \beta_B} f(\alpha_B + 1, \beta_B, \alpha_A, \beta_A),

where :math:`f(\alpha_B, \beta_B, \alpha_A, \beta_A)` is :math:`P[X_A > X_B]` and :math:`f(\alpha_B, \beta_B, \alpha_A, \beta_A)` is :math:`P[X_B > X_A]`. A similar expression is obtained for :math:`\mathrm{EL}(X_A)`,

.. math::

   \mathrm{EL}(X_A) = \frac{\alpha_B}{\alpha_B + \beta_B} f(\alpha_A, \beta_A, \alpha_B + 1, \beta_B) - \frac{\alpha_A}{\alpha_A + \beta_A} f(\alpha_A + 1, \beta_A, \alpha_B, \beta_B).

See also :cite:`Stucchio2014`.

The credible intervals
----------------------

Credible intervals are employed to account for uncertainty in the expected loss and relative expected loss measures. Let us considered the relative expected loss if variant B is chosen, which follows the distribution :math:`(X_A - X_B)/X_B = X_A / X_B - 1`. This requires the distribution of the ratio of two random beta variables, :math:`U = X_A / X_B`. The probability density function is given by

.. math::

   f(u) = \int_0^{\infty} x f(x, u x) \mathop{dx},

where :math:`f(x, u x)` is the joint probability distribution. After a few operations we obtain

.. math::

   x f(x,u x) &= x \frac{x^{\alpha_A - 1} (1-x)^{\beta_A - 1}}{B(\alpha_A, \beta_A)} \frac{(ux)^{\alpha_B - 1} (1-ux)^{\beta_B - 1}}{B(\alpha_B, \beta_B)},\\
   &= \frac{u^{\alpha_B - 1}}{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)} x^{\alpha_A + \alpha_B - 1}(1-x)^{\beta_B - 1}(1-ux)^{\beta_A - 1}.

We divide the domain of computation. First, we consider the case :math:`u \in (0, 1)`, obtaining the probability density function

.. math::

   f(u) &= \frac{u^{\alpha_B - 1}}{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)} \int_0^1 x^{\alpha_A + \alpha_B - 1}(1-x)^{\beta_B - 1}(1-ux)^{\beta_A - 1} \mathop{dx}\\
   &= \frac{B(\alpha_A + \alpha_B, \beta_B) u^{\alpha_B - 1}}{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)}\, _2F_1(1-\beta_A, \alpha_A + \alpha_B; \alpha_A + \alpha_B + \beta_B; u)\\
   &= \frac{B(\alpha_A + \alpha_B, \beta_B) u^{\alpha_B - 1} (1-u)^{\beta_A + \beta_B - 1}}{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)}\, _2F_1(\beta_B, \alpha_A + \alpha_B + \beta_A + \beta_B - 1; \alpha_A + \alpha_B + \beta_B; u).

The cumulative density function can be computed as follows

.. math::

   F(u) = C \int_0^u t^{\alpha_B - 1} (1-t)^{\beta_A + \beta_B - 1}\, _2F_1(\beta_B, \alpha_A + \alpha_B + \beta_A + \beta_B - 1; \alpha_A + \alpha_B + \beta_B; t) \mathop{dt},

where

.. math::

   C = \frac{B(\alpha_A + \alpha_B, \beta_B) }{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)}

Thus,

.. math::

   F(u) &= C \sum_{k=0}^{\infty} \frac{(\beta_B)_k (\alpha_A + \alpha_B + \beta_A + \beta_B - 1, k)}{(\alpha_A + \alpha_B + \beta_B)_k k!}\int_0^u t^{\alpha_A - 1 + k} (1-t)^{\beta_A + \beta_B - 1}\mathop{dt}\\
   &= C \sum_{k=0}^{\infty} \frac{(\beta_B)_k (\alpha_A + \alpha_B + \beta_A + \beta_B - 1, k)}{(\alpha_A + \alpha_B + \beta_B)_k} \frac{B_u (\alpha_A + k, \beta_A + \beta_B)}{k!},\\
   &= C  \frac{u^{\alpha_A}}{\alpha_A} \, _3F_2\left({\alpha_A, \alpha_A + \alpha_B, 1 - \beta_A \atop \alpha_A + 1, \alpha_A + \alpha_B + \beta_B};u\right),

where :math:`B_z(a, b)` is the incomplete beta function. Similarly, for :math:`u \in (1, \infty)` we have the probability density function

.. math::

   f(u) & = C \left(\frac{1}{u}\right)^{\alpha_B + 1} \, _2F_1 \left(\alpha_A + \alpha_B, 1 -\beta_B; \alpha_A + \alpha_B + \beta_A; \frac{1}{u}\right)\\
   &= C \left(\frac{1}{u}\right)^{\alpha_B + 1} \left(1- \frac{1}{u}\right)^{\beta_A + \beta_B - 1} \, _2F_1 \left(\beta_A, \alpha_A + \alpha_B + \beta_A + \beta_B - 1; \alpha_A + \alpha_B + \beta_A; \frac{1}{u}\right),

where constant :math:`C` is defined by 

.. math::

   C = \frac{B(\alpha_A + \alpha_B, \beta_A)}{B(\alpha_A, \beta_A) B(\alpha_B, \beta_B)}.

The cumulative density function is given by

.. math::

   F(u) &= 1 - C \sum_{k=0}^{\infty} \frac{(\beta_A)_k (\alpha_A + \alpha_B + \beta_A + \beta_B - 1)_k}{(\alpha_A + \alpha_B + \beta_A)_k} \frac{B_{1/u}(\alpha_B + k, \beta_A + \beta_B)}{k!},\\
   &= 1 - C \frac{u^{-\alpha_B}}{\alpha_B}\, _3F_2\left({\alpha_B, \alpha_A + \alpha_B, 1 - \beta_B \atop \alpha_B + 1, \alpha_A + \alpha_B + \beta_A}; \frac{1}{u}\right).

.. note::

   Credible intervals are computed by solving :math:`F(u) = p`, :math:`p \in [0, 1]`. A reasonable starting point is the normal approximation of the beta distribution.


The expected value and variance of the distribution :math:`Z = (X_A - X_B)/X_B = X_A / X_B - 1` can be computed using

.. math::

   \mathrm{E}\left[\frac{X_A}{X_B} \right] = \frac{\alpha_A (\alpha_B + \beta_B - 1)}{(\alpha_A + \beta_A)(\alpha_B - 1)}.

.. math::

   \mathrm{Var} \left[\frac{X_A}{X_B} \right] = \mathrm{E}\left[\frac{X_A}{X_B} \right] \left(\frac{(\alpha_A + 1) (\alpha_B + \beta_B - 2)}{(\alpha_A + \beta_A + 1)(\alpha_B - 2)} - \mathrm{E}\left[\frac{X_A}{X_B} \right]\right).


Proofs
------

The error probability
"""""""""""""""""""""

Integrating the joint distribution, under the assumption of independence, over all values of :math:`X_B > X_A` we obtain the following integral

.. math::

   P[X_B > X_A] &= \int_0^1 \int_{x_A}^1 \frac{x_A^{\alpha_A - 1} (1-x_A)^{\beta_A - 1}}{B(\alpha_A, \beta_A)}\frac{x_B^{\alpha_B - 1} (1-x_B)^{\beta_B - 1}}{B(\alpha_B,  \beta_B)}\mathop{dx_B} \mathop{dx_A}\\
   &= 1 - \int_0^1 \frac{x_A^{\alpha_A - 1} (1-x_A)^{\beta_A - 1}}{B(\alpha_A,  \beta_A)} I_{x_A}(\alpha_B, \beta_B) \mathop{dx_A},

where :math:`I_x(a,b)` is the regularized incomplete beta function defined by

.. math::

   I_x(a,b) = \frac{x^a}{B(a,b)}\sum_{k=0}^{\infty} \frac{(1-b)_k}{(a+k)}\frac{x^k}{k!}, \quad |x| < 1.

Let us focus on the above integral :math:`I`. By formally interchanging integration and summation, which is justified by the absolute convergence of the hypergeometric series, we obtain

.. math::

   I &= \int_0^1 \frac{x^{\alpha_A - 1} (1-x)^{\beta_A - 1} x^{\alpha_B}}{B(\alpha_A, \beta_A)B(\alpha_B, \beta_B)} \sum_{k=0}^{\infty} \frac{(1-\beta_B)_k}{(\alpha_B + k)}\frac{x^k}{k!} \mathop{dx}\\
   &= \frac{1}{B(\alpha_A, \beta_A)B(\alpha_B, \beta_B)} \sum_{k=0}^{\infty} \frac{(1-\beta_B)_k}{(\alpha_B + k) k!}\int_0^1 x^{\alpha_A + \alpha_B + k - 1}(1-x)^{\beta_A-1} \mathop{dx}\\
   &= \frac{\Gamma(\beta_A)}{B(\alpha_A, \beta_A)B(\alpha_B, \beta_B)} \sum_{k=0}^{\infty} \frac{(1-\beta_B)_k}{(\alpha_B + k) k!} \frac{\Gamma(\alpha_A + \alpha_B + k)}{\Gamma(\alpha_A + \alpha_B +\beta_A + k)}.

Finally, note that the resulting series is hypergeometric and expressible in terms of :math:`_3F_2` which yields

.. math::

   \sum_{k=0}^{\infty} \frac{(1-\beta_B)_k}{(\alpha_B + k) k!} \frac{\Gamma(\alpha_A + \alpha_B + k)}{\Gamma(\alpha_A + \alpha_B +\beta_A + k)} = \frac{1}{\alpha_B (\alpha_A+\alpha_B)_{\beta_A}}\,_3F_2\left({\alpha_B, \alpha_A+\alpha_B, 1-\beta_B \atop 1+\alpha_B, \alpha_A + \alpha_B + \beta_A};1\right).

Rearranging terms the proof is completed. 


The relative expected loss function
"""""""""""""""""""""""""""""""""""

The moments of :math:`Z = X_A / X_B` are given by

.. math::

   \frac{B(\alpha_A + k, \beta_A) B(\alpha_B - k, \beta_B)}{B(\alpha_A, \beta_A)B(\alpha_B, \beta_B)}.

Therefore, the expected value or moment :math:`k=1`, is

.. math::

   \frac{B(\alpha_A + 1, \beta_A) B(\alpha_B - 1, \beta_B)}{B(\alpha_A, \beta_A)B(\alpha_B, \beta_B)} = \frac{\alpha_A (\alpha_B + \beta_B - 1)}{(\alpha_A + \beta_A)(\alpha_B - 1)}.

Note that the second moment, :math:`k=2`, is required to compute the variance.


References
----------

.. bibliography:: refs.bib
   :filter: docname in docnames