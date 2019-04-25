General formulas and definitions
================================

The main metrics to perform A/B testing are described in :cite:`Stucchio2015`. Let us consider two variants :math:`X_A` and :math:`X_B` for testing.

The **error probability** or probability of :math:`X_B > X_A` is denoted as

.. math::
   P[X_B > X_A] = \int_{-\infty}^{\infty} \int_{x_A}^{\infty} f(x_A, x_B) \mathop{dx_B} \mathop{dx_A},

where :math:`f(x_A, x_B)` is the joint probability distribution, under the assumption of independence, i.e. :math:`f(x_A, x_B) = f(x_A) f(x_B)`.

The **expected loss function** given a joint posterior is the expected value of the **loss function**. The loss function is the expected uplift lost by choosing a given variant. If variant :math:`X_B` is chosen we have

.. math::

   EL(X_B) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max(x_A - x_B, 0) f(x_A, x_B) \mathop{dx_B} \mathop{dx_A}.

Other metrics also considered are the **relative expected loss** or uplift and credible intervals. A credible interval is a region which has a specified probability of containing the true value.

References
----------

.. bibliography:: refs.bib
   :filter: docname in docnames
