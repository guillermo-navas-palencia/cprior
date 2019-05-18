"""
Base A/B testing experiment class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import time

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
        method="exact", min_samples=1000, max_samples=10000, max_time=1e6,
        verbose=True):
        self.abtest = abtest
        self.stopping_rule = stopping_rule
        self.epsilon = epsilon
        self.method = method
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.max_time = max_time
        self.verbose = verbose

        self._threshold = self.epsilon

        # auxiliary data / statistics
        self._measure_A = []
        self._measure_B = []

        self._samples_A = []
        self._samples_B = []
        self._sum_A = []
        self._sum_B = []

        # timing
        self._time_init = None
        self._time_termination = None

        # flags
        self._status = None
        self._termination = False

        self._setup()

    def results(self):
        """
        """
        if not self._termination:
            raise ValueError("not finished!")

        if self.verbose and self._status is None:
            pass

    def stats(self):
        """
        """
        if not self._termination:
            raise ValueError("not finished!")

        if self.verbose and self._status is None:
            pass

    def update(self, dataA=None, dataB=None):
        """
        Update posterior parameters for one or both variants with new data
        samples.

        Parameters
        ----------
        dataA : array-like, shape = (n_samples) (default=None)
            Data samples for modelA.

        dataB : array-like, shape = (n_samples) (default=None)
            Data samples for modelB.
        """
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
        min_nAB = min(n_A, n_B)

        max_samples_exceeded = min_nAB > self.max_samples
        min_samples_exceeded = min_nAB > self.min_samples

        if self.stopping_rule == "expected_loss":
            variant_A = self.abtest.expected_loss(variant="A",
                method=self.method)
            variant_B = self.abtest.expected_loss(variant="B",
                method=self.method)
        elif self.stopping_rule == "error_probability":
            variant_A = self.abtest.probability(variant="A", method=self.method)
            variant_B = self.abtest.probability(variant="B", method=self.method)

        self._measure_A.append(variant_A)
        self._measure_B.append(variant_B)

        if min_samples_exceeded:
            if variant_A < self._threshold
                self._status = "A winner"
                self._termination = True
            elif variant_B < self._threshold:
                self._status = "B winner"
                self._termination = True
            elif max_samples_exceeded:
                self._status = "fail"
                self._termination = True

    def _setup(self):
        # input checks

        if self.stopping_rule == "expected_loss":
            self._threshold = self.epsilon
        elif self.stopping_rule == "error_probability":
            self._threshold = 1.0 - self.epsilon

    @property
    def termination(self):
        return self._termination
