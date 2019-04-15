#include <cmath>
#include <iostream>

#include "cprior.hpp"

double betaln(double a, double b)
{
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

double beta_terminating_a0(int a0, int b0, int a1, int b1, double err=1e-15)
{
  return 0;
}

double beta_terminating_a1(int a0, int b0, int a1, int b1, double err=1e-15)
{
  double c, s, sp;
  const int a01 = a0 - 1;
  const int ab01 = a0 + b0 + b1 - 1;

  c = std::exp(betaln(a0 + a1 - 1, b0 + b1) - betaln(a1, b1) - betaln(a0, b0));
  s = c / (b1 + a1 - 1);
  sp = s;
  for (int k = a1 - 1; k >= 0; k--) {
    c *= static_cast<double>(ab01 + k) * k / (a01 + k) / (k + b1);
    s += c / (b1 + k - 1);
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return s;
}

double beta_terminating_b0(int a0, int b0, int a1, int b1, double err=1e-15)
{
  return 0;
}

double beta_terminating_b1(int a0, int b0, int a1, int b1, double err=1e-15)
{
  return 0;
}

double beta_3f2(double a0, double b0, double a1, double b1, double err=1e-15)
{
  const int maxiter = 10000;
  const double aa = a0 + b0;
  const double bb = b0 + b1;
  const double cc = b0 + 1;
  const double dd = a0 + b0 + a1 + b1;

  double c, s, sp, t;

  c = std::exp(betaln(a0+a1,bb)-(std::log(b0) + betaln(a0,b0) + betaln(a1,b1)));
  t = c;
  s = t;
  sp = s;
  for (int k = 0; k <= maxiter; k++) {
    t *= (aa + k) * (bb + k) / ((cc + k) * (dd + k));
    s += t;
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return 1.0 - s;
}

double beta_asymptotic(int a0, int b0, int a1, int b1)
{
  return 0;
}

double beta_cprior(int a0, int b0, int a1, int b1)
{
  // calculate min value a0, b0, a1, b1 and implement special case for each one.

  if ((a1 + b1 >= 2.0 * (a0 + b0)) and (a1 > b1) and (a1 > 300))
    return beta_3f2(a0, b0, a1, b1);
  else
    return beta_terminating_a1(a0, b0, a1, b1);
}