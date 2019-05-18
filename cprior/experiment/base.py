"""
Base A/B testing experiment class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np


class ExperimentAB(object):
    """
    
    Parameters
    ----------
    abtest : object

    stopping_rule : str (default="expected_loss")

    epsilon : float (default=1e-5)

    min_samples : int (default=1000)

    verbose: boolean or int (default=True)
    """
    def __init__(self, abtest, stopping_rule="expected_loss", epsilon=1e-5,
        min_samples=1000, verbose=True):
        self.abtest = abtest
        self.stopping_rule = stopping_rule
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.verbose = verbose

        self._measure_A = []
        self._measure_B = []

        self._samples_A = []
        self._samples_B = []
        self._sum_A = []
        self._sum_B = []

        self._termination = False

    def update(self, dataA=None, dataB=None):
        if dataA is not None:
            x = np.asarray(dataA)
            self.abtest.update_A(x)
            self._samples_A.append(x.size)
            self._sum_A.append(np.sum(x))
        if dataB is not None:
            x = np.asarray(dataB)
            self.abtest.update_B(x)
            self._samples_B.append(x.size)
            self._sum_B.append(np.sum(x))

        if (dataA is not None) or (dataB is not None):
            self._compute_stopping_rule()

    def _compute_stopping_rule(self):
        
        n_A = self.abtest.modelA.n_samples_
        n_B = self.abtest.modelB.n_samples_
        termination_n_AB = min(n_A, n_B) > self.min_samples

        if self.stopping_rule == "expected_loss":
            variant_A = self.abtest.expected_loss(variant="A")
            variant_B = self.abtest.expected_loss(variant="B")

            self._measure_A.append(variant_A)
            self._measure_B.append(variant_B)

            if variant_A < self.epsilon and termination_n_AB:
                self._winner = "A"
                self._termination = True
            elif variant_B < self.epsilon and termination_n_AB:
                self._winner = "B"
                self._termination = True

        elif self.stopping_rule == "error_probability":
            variant_A = self.abtest.probability(variant="A")
            variant_B = self.abtest.probability(variant="B")

            self._measure_A.append(variant_A)
            self._measure_B.append(variant_B)

            if variant_A > self.epsilon and termination_n_AB:
                self._winner = "A"
                self._termination = True
            elif variant_B > self.epsilon and termination_n_AB:
                self._winner = "B"
                self._termination = True       

    def stats(self):
        if not self._termination:
            raise ValueError("not finished!")

    @property
    def termination(self):
        return self._termination
