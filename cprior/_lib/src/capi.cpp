#include "cprior.hpp"
#include "cprior.h"


double cpp_beta_cprior(double a0, double b0, double a1, double b1)
{
  return beta_cprior(a0, b0, a1, b1);
}