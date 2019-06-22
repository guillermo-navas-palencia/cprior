"""
Example Bayesian model with Bernoulli distribution.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import scipy.stats as st

from cprior import BernoulliModel
from cprior import BernoulliABTest

# Two model variants A and B, and build A/B test
modelA = BernoulliModel(alpha=1, beta=1)
modelB = BernoulliModel(alpha=1, beta=1)

test = BernoulliABTest(modelA=modelA, modelB=modelB, simulations=1000000)

# Generate new data and update models
data_A = st.bernoulli(p=0.10).rvs(size=1500, random_state=42)
data_B = st.bernoulli(p=0.11).rvs(size=1600, random_state=42)

test.update_A(data_A)
test.update_B(data_B)

# Compute P[A > B] and P[B > A]
print("P[A > B] = {:.4f}".format(test.probability(variant="A")))
print("P[B > A] = {:.4f}".format(test.probability(variant="B")))

# Compute posterior expected loss given a variant
print("E[max(B - A, 0)] = {:.4f}".format(test.expected_loss(variant="A")))
print("E[max(A - B, 0)] = {:.4f}".format(test.expected_loss(variant="B")))
