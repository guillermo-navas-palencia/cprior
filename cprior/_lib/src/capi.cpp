#include "cprior.hpp"
#include "cprior.h"


double cpp_beta_cprior(int a0, int b0, int a1, int b1)
{
	return beta_cprior(a0, b0, a1, b1);
}