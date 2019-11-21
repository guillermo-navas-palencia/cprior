#include "cprior.hpp"
#include "cprior.h"


double cpp_beta_cprior(double a0, double b0, double a1, double b1)
{
  return beta_cprior(a0, b0, a1, b1);
}

double cpp_beta_binomial_cdf_cprior(int k, int n, double a, double b)
{
  return beta_binomial_cdf_cprior(k, n, a, b);
}