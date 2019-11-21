#ifndef CPRIOR_H
#define CPRIOR_H

double betaln(double a, double b);
double beta_cprior(double a0, double b0, double a1, double b1);

double beta_binomial_cdf_cprior(int k, int n, double a, double b);

#endif /* CPRIOR_H */