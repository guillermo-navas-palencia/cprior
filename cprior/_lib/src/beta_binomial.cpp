/*
  Closed-form formulas for beta-binomial distribution.

  Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
  Copyright (C) 2019
*/

#include <cmath>

#include "cprior.hpp"

double combln(int n, int k)
{
  return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1); 
}

double beta_binomial_cdf_cprior(int k, int n, double a, double b)
{
  const double err = 1e-15;
  const double bn = b + n;
  const double a1 = a - 1;
  const double n1 = n + 1;

  double c, s, sp;

  c = std::exp(combln(n, k) + betaln(a + k, b + n - k) - betaln(a, b));
  s = c;
  sp = s;
  for (int j = k; j >= 0; j--) {
    c *= j * (bn - j) / ((a1 + j) * (n1 - j));
    s += c;
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return s;
}

int beta_binomial_ppf_cprior(double q, int n, double a, double b)
{
  int k = 0;

  while (q > beta_binomial_cdf_cprior(k, n, a, b))
    k++;

  return k;
}