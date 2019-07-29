Pareto distribution
===================

Error probability or chance to beat
-----------------------------------

Given two distributions :math:`X_A \sim \mathcal{PA}(\alpha_A, \beta_A)` and :math:`X_B \sim \mathcal{PA}(\alpha_B, \beta_B)` such that :math:`(\alpha_A, \beta_A, \alpha_B, \beta_B) \in \mathbb{R}_+^4`, :math:`P[X_B > X_A]` is given by

.. math::

   P[X_B > X_A] = \begin{cases}
      1 - \frac{\alpha_B}{\alpha_A + \alpha_B}\left(\frac{\beta_A}{\beta_B}\right)^{\alpha_A}, & \beta_B > \beta_A,\\
      \frac{\alpha_A}{\alpha_A + \alpha_B}\left(\frac{\beta_B}{\beta_A}\right)^{\alpha_B}, & \beta_B \le \beta_A
   \end{cases}


Expected loss function
----------------------

The expected loss function can easily be calculated from the `definition <formulas_conjugate_general.html>`__ yielding

.. math::

   \mathrm{EL}(X_B) = \begin{cases}
      \frac{\alpha_A \beta_A}{\alpha_A - 1}\left(1- \left(\frac{\beta_B}{\beta_A}\right)^{\alpha_B}\right) - \frac{\alpha_B}{\alpha_B - 1}\left(\beta_B - \beta_A \left(\frac{\beta_B}{\beta_A}\right)^{\alpha_B}\right) + \frac{\alpha_B \beta_A \left(\frac{\beta_B}{\beta_A}\right)^{\alpha_B} }{(\alpha_B + \alpha_A - 1)(\alpha_A - 1)}, & \beta_A > \beta_B\\
      \frac{\alpha_B \beta_B}{(\alpha_B + \alpha_A - 1)(\alpha_A - 1)}\left(\frac{\beta_A}{\beta_B}\right)^{\alpha_A}, & \beta_A \le \beta_B
   \end{cases}


and similarly for :math:`\mathrm{EL}(X_A)`,

.. math::

   \mathrm{EL}(X_A) = \begin{cases}
      \frac{\alpha_B \beta_B}{\alpha_B - 1}\left(1- \left(\frac{\beta_A}{\beta_B}\right)^{\alpha_A}\right) - \frac{\alpha_A}{\alpha_A - 1}\left(\beta_A - \beta_B \left(\frac{\beta_A}{\beta_B}\right)^{\alpha_A}\right) + \frac{\alpha_A \beta_B \left(\frac{\beta_A}{\beta_B}\right)^{\alpha_A} }{(\alpha_A + \alpha_B - 1)(\alpha_B - 1)}, & \beta_B > \beta_A\\
      \frac{\alpha_A \beta_A}{(\alpha_A + \alpha_B - 1)(\alpha_B - 1)}\left(\frac{\beta_B}{\beta_A}\right)^{\alpha_B},  & \beta_B \le \beta_A
   \end{cases}

Credible intervals
------------------

The expected value of the distribution :math:`Z = (X_A - X_B)/X_B = X_A / X_B - 1` can be computed using

.. math::

   \mathrm{E}\left[\frac{X_A}{X_B}\right] = \frac{\alpha_A}{\alpha_A - 1} \frac{\alpha_B \beta_A}{(\beta_B (\alpha_B + 1))}.

Proofs
------

Error probability
"""""""""""""""""

Integrating the joint distribution over all values of :math:`X_B > X_A` we obtain the integral

.. math::

   P[X_B > X_A] &= \int_{\beta_A}^{\infty} \int_{\max(\beta_B, x_A)}^{\infty} \frac{\alpha_A \beta_A^{\alpha_A}}{x_A^{\alpha_A + 1}} \frac{\alpha_B \beta_B^{\alpha_B}}{x_B^{\alpha_B + 1}} \mathop{dx_B} \mathop{dx_A}\\
   &= 1 - \int_{\beta_A}^{\infty} \frac{\alpha_A \beta_A^{\alpha_A}}{x_A^{\alpha_A + 1}} \left(1 - \left(\frac{\beta_B}{\max(\beta_B, x_A)}\right)^{\alpha_B}\right) \mathop{dx_A}\\
   &= \alpha_A \beta_A^{\alpha_A} \beta_B^{\alpha_B} \int_{\beta_A}^{\infty} \frac{1}{x^{\alpha_A + 1} \max(\beta_B, x)^{\alpha_B}} \mathop{dx}.


Case :math:`\beta_B > \beta_A`:

.. math::

   \int_{\beta_A}^{\infty} \frac{1}{x^{\alpha_A + 1} \max(\beta_B, x)^{\alpha_B}} \mathop{dx} &= \int_{\beta_A}^{\beta_B} x^{-\alpha_A - 1} \beta_B^{-\alpha_B} \mathop{dx} + \int_{\beta_B}^{\infty} x^{-\alpha_A - 1} x^{-\alpha_B} \mathop{dx}\\
   &= \beta_B^{-\alpha_B} \left(\frac{\beta_A^{-\alpha_A} - \beta_B^{-\alpha_A}}{\alpha_A}\right) + \frac{\beta_B^{-\alpha_A - \alpha_B}}{\alpha_A + \alpha_B}.

.. math::

   P[X_B > X_A] &= \alpha_A \beta_A^{\alpha_A} \beta_B^{\alpha_B} \left[\beta_B^{-\alpha_B} \left(\frac{\beta_A^{-\alpha_A} - \beta_B^{-\alpha_A}}{\alpha_A}\right) + \frac{\beta_B^{-\alpha_A - \alpha_B}}{\alpha_A + \alpha_B}\right]\\
   &= 1 - \frac{\alpha_B}{\alpha_A + \alpha_B}\left(\frac{\beta_A}{\beta_B}\right)^{\alpha_A}.


Case :math:`\beta_B \le \beta_A`:

.. math::

   \int_{\beta_A}^{\infty} \frac{1}{x^{\alpha_A + 1} \max(\beta_B, x)^{\alpha_B}} \mathop{dx} = \int_{\beta_A}^{\infty} x^{-\alpha_A - \alpha_B - 1} \mathop{dx}
   = \frac{\beta_A^{-\alpha_A - \alpha_B}}{\alpha_A + \alpha_B}

.. math::

   P[X_B > X_A] = \alpha_A \beta_A^{\alpha_A} \beta_B^{\alpha_B} \frac{\beta_A^{-\alpha_A - \alpha_B}}{\alpha_A + \alpha_B} = \frac{\alpha_A}{\alpha_A + \alpha_B}\left(\frac{\beta_B}{\beta_A}\right)^{\alpha_B}.
