Gamma distribution
==================

Error probability or chance to beat
-----------------------------------

Given two distributions :math:`X_A \sim \mathcal{G}(\alpha_A, \beta_A)` and :math:`X_B \sim \mathcal{G}(\alpha_B, \beta_B)` such that :math:`(\alpha_A, \beta_A, \alpha_B, \beta_B) \in \mathbb{R}_+^4`, :math:`P[X_B > X_A]` is given by

.. math::

   P[X_B > X_A] = 1 - \frac{\beta_A^{\alpha_A}\beta_B^{\alpha_B}}{(\beta_A + \beta_B)^{\alpha_A+\alpha_B}}\frac{_2F_1\left(1, \alpha_A + \alpha_B; \alpha_B + 1; \frac{\beta_B}{\beta_B + \beta_A}\right)}{\alpha_B B(\alpha_A, \alpha_B)} = I_{\frac{\beta_A}{\beta_A + \beta_B}}(\alpha_A, \alpha_B),

where :math:`_2F_1(a,b;c;z)` is the Gauss hypergeometric function and :math:`I_x(a,b)` is the regularized incomplete beta function.

Expected loss function
----------------------

Credible intervals
------------------


Proofs
------

Error probability
"""""""""""""""""

Integrating the joint distribution over all values of :math:`X_B > X_A` we obtain the integral

.. math::
   P[X_B > X_A] &= \int_0^{\infty} \int_{x_A}^{\infty} \frac{\beta_A^{\alpha_A}}{\Gamma(\alpha_A)} x_A^{\alpha_A - 1} e^{-\beta_A x_A} \frac{\beta_B^{\alpha_B}}{\Gamma(\alpha_B)} x_B^{\alpha_B - 1} e^{-\beta_B x_A} \mathop{dx_B}\mathop{dx_A}\\
   &= 1 - \int_0^{\infty}\frac{\beta_A^{\alpha_A}}{\Gamma(\alpha_A)} x_A^{\alpha_A - 1} e^{-\beta_A x_A} P(\alpha_B, \beta_B x_A)\mathop{dx_A},

where :math:`P(a,z)` is the regularized lower incomplete gamma function defined by

.. math::
   
   P(a, z) = \frac{\gamma(a, z)}{\Gamma(a)} = 1 - Q(a,z),

and :math:`Q(a,z)` is the regularized upper incomplete gamma function with series expansion

.. math::

   Q(a,z) = \frac{\Gamma(a,z)}{\Gamma(a)} = 1 - z^a e^{-z} \sum_{k=0}^{\infty}\frac{z^k}{\Gamma(a+k+1)}.

Hence, the integral is rewritten in the form

.. math::

   P[X_B > X_A] = \int_0^{\infty}\frac{\beta_A^{\alpha_A}}{\Gamma(\alpha_A)} x^{\alpha_A - 1} e^{-\beta_A x} Q(\alpha_B, \beta_B x) \mathop{dx}.

Interchange of integration and summation leads to a representation in terms of the Gauss hypergeometric function :math:`_2F_1(a,b;c;z)`, also expressible in terms of the incomplete beta function

.. math::

   P[X_B > X_A] &= 1 - \frac{\beta_A^{\alpha_A}\beta_B^{\alpha_B}}{\Gamma(\alpha_A)}\int_0^{\infty} x^{\alpha_A + \alpha_B - 1} e^{-(\beta_A + \beta_B) x} \sum_{k=0}^{\infty}\frac{(\beta_B x)^k}{\Gamma(\alpha_B + k + 1)}  \mathop{dx}\\
   &= 1 - \frac{\beta_A^{\alpha_A}\beta_B^{\alpha_B}}{\Gamma(\alpha_A)}
   \sum_{k=0}^{\infty} \frac{\beta_B^k}{\Gamma(\alpha_B + k + 1)}\int_0^{\infty}x^{\alpha_A + \alpha_B + k - 1} e^{-(\beta_A + \beta_B) x}\mathop{dx}\\
   & =1 - \frac{\beta_A^{\alpha_A}\beta_B^{\alpha_B}}{(\beta_A + \beta_B)^{\alpha_A+\alpha_B}}\sum_{k=0}^{\infty}\frac{\Gamma(\alpha_A + \alpha_B + k)}{\Gamma(\alpha_B + k + 1) \Gamma(\alpha_A)} \left(\frac{\beta_B}{\beta_A + \beta_B}\right)^k\\
   &=1 - \frac{\beta_A^{\alpha_A}\beta_B^{\alpha_B}}{(\beta_A + \beta_B)^{\alpha_A+\alpha_B}}\frac{_2F_1\left(1, \alpha_A + \alpha_B; \alpha_B + 1; \frac{\beta_B}{\beta_B + \beta_A}\right)}{\alpha_B B(\alpha_A, \alpha_B)}= I_{\frac{\beta_A}{\beta_A + \beta_B}}(\alpha_A, \alpha_B).
