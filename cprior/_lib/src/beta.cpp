/*
  Closed-form formulas for Beta distribution.

  Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
  Copyright (C) 2019
*/

#include <algorithm>
#include <cmath>

#include "cprior.hpp"

double betaln(double a, double b)
{
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

double beta_terminating_a0(int a0, double b0, double a1, double b1,
  double err=1e-15)
{
  long double c, s, sp;
  const long double a11 = a1 - 1;
  const long double ab11 = a1 + b0 + b1 - 1;

  c = std::exp((long double)(betaln(a1 + a0 - 1, b0 + b1) - betaln(a1, b1) - betaln(a0, b0)));
  if (c == 0)
    return nan("");


  s = c / (b0 + a0 - 1);
  sp = s;
  for (int k = a0 - 1; k >= 0; k--) {
    c *= (ab11 + k) * k / (a11 + k) / (k + b0);
    s += c / (b0 + k - 1);
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return 1.0 - s;
}

double beta_terminating_a1(double a0, double b0, int a1, double b1,
  double err=1e-15)
{
  long double c, s, sp;
  const long double a01 = a0 - 1;
  const long double ab01 = a0 + b0 + b1 - 1;

  c = std::exp((long double)(betaln(a0 + a1 - 1, b0 + b1) - betaln(a1, b1) - betaln(a0, b0)));
  if (c == 0)
    return nan("");

  s = c / (b1 + a1 - 1);
  sp = s;
  for (int k = a1 - 1; k >= 0; k--) {
    c *= (ab01 + k) * k / (a01 + k) / (k + b1);
    s += c / (b1 + k - 1);
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return s;
}

double beta_terminating_b0(double a0, int b0, double a1, double b1,
  double err=1e-15)
{
  long double c, s, sp;
  const long double b11 = b1 - 1;
  const long double ab11 = b1 + a0 + a1 - 1;

  c = std::exp((long double)(betaln(b1 + b0 - 1, a0 + a1) - betaln(a1, b1) - betaln(a0, b0)));
  if (c == 0)
    return nan("");

  s = c / (a0 + b0 - 1);
  sp = s;
  for (int k = b0 - 1; k >= 0; k--) {
    c *= (ab11 + k) * k / (b11 + k) / (k + a0);
    s += c / (a0 + k - 1);
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return s;
}

double beta_terminating_b1(double a0, double b0, double a1, int b1,
  double err=1e-15)
{
  long double c, s, sp;
  const long double b01 = b0 - 1;
  const long double ab01 = b0 + a0 + a1 - 1;

  c = std::exp((long double)(betaln(b0 + b1 - 1, a0 + a1) - betaln(a1, b1) - betaln(a0, b0)));
  if (c == 0)
    return nan("");

  s = c / (a1 + b1 - 1);
  sp = s;
  for (int k = b1 - 1; k >= 0; k--) {
    c *= (ab01 + k) * k / (b01 + k) / (k + a1);
    s += c / (a1 + k - 1);
    if (std::abs((s - sp) / s) < err)
      break;
    else
      sp = s;
  }
  return 1.0 - s;
}

double beta_3f2(double a0, double b0, double a1, double b1, double err=1e-15)
{
  const int maxiter = 10000;
  const long double aa = a0 + b0;
  const long double bb = b0 + b1;
  const long double cc = b0 + 1;
  const long double dd = a0 + b0 + a1 + b1;

  long double c, s, sp, t;

  c = std::exp((long double)(betaln(a0+a1,bb)-(std::log(b0) + betaln(a0,b0) + betaln(a1,b1))));
  if (c == 0)
    return nan("");

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

double beta_cprior(double a0, double b0, double a1, double b1)
{
  long double int_a0, int_a1, int_b0, int_b1;

  if (a0 <= std::min(std::min(b0, b1), a1) && std::modf(a0, &int_a0) == 0.0)
    return beta_terminating_a0((int)a0, b0, a1, b1);
  else if (b0 <= std::min(std::min(a0, a1), b1) && std::modf(b0, &int_b0) == 0.0)
    return beta_terminating_b0(a0, (int)b0, a1, b1);
  else if (a1 <= std::min(std::min(a0, b0), b1) && std::modf(a1, &int_a1) == 0.0)
    return beta_terminating_a1(a0, b0, (int)a1, b1);
  else if (b1 <= std::min(std::min(a0, a1), b0) && std::modf(b1, &int_b1) == 0.0)
    return beta_terminating_b1(a0, b0, a1, (int)b1);
  else
    return beta_3f2(a0, b0, a1, b1);
}