Utilities for Bayesian models and distributions
===============================================

Confidence/credible intervals
-----------------------------

Credible intervals quantify the uncertainty of a parameter by providing the
range of values containing the true parameter value with a given probability.
Credible intervals can be calculated using the equal-tailed quantile method
(ETI) or the highest posterior density method (HDI).

The HDI method calculates the interval such that
:math:`P(l < z < u) = 1 - \delta`, where :math:`\delta` denotes the significant
level, and :math:`l` and :math:`u` denote the lower and upper bound of the
interval, respectively. The HDI computes the narrowest interval by solving
the minimization problem, see :cite:`Chen1999`

.. math::

   \underset{l < u}{\text{min}}\left(|f(u) - f(l)| + |F(u) - F(l) - (1 -\delta)|
   \right).

We reformulate the problem by removing absolute values and adding the narrowest
interval :math:`u - l` on the objective function,

.. math::

   \underset{u,l, t, w}{\text{min}} &\quad t + w + u - l\\
   \text{s.t.} &\quad  -t + f(u) - f(l) \ge 0\\
   &\quad t + f(u) - f(l) \ge 0\\
   &\quad -w + F(u) - F(l) - (1 -\delta) \ge 0\\
   &\quad w + F(u) - F(l) - (1 -\delta) \ge 0\\
   &\quad u - l - \epsilon \ge 0\\
   &\quad l \in [l_{\min}, l_{\max}]]\\
   &\quad u \in [u_{\min}, u_{\max}]

where :math:`\epsilon > 0`, say :math:`\epsilon = 0.000001`. Parameters
:math:`l_{\min}`, :math:`l_{\max}`, :math:`u_{\min}` and :math:`u_{\max}`
denote the bounds for the interval limits :math:`l` and :math:`u`, there are
dependent on the statistical distribution support.


.. autofunction:: cprior.cdist.ci_interval


.. autofunction:: cprior.cdist.ci_interval_exact


References
----------

.. bibliography:: refs.bib
   :filter: docname in docnames