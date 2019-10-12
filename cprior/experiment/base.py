"""
Bayesian A/B and MV testing experiment.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import copy
import numbers
import time

import numpy as np

from ..cdist.base import BayesMVTest

_STATUS_MAX_SAMPLES = "max_samples exceeded"
_STATUS_RUNNING = "running..."
_STATUS_WINNER = "winner {}"

_STOPPING_RULES = ["probability", "expected_loss", "expected_loss_vs_all",
                   "probability_vs_all"]


class Experiment(object):
    """
    Bayesian experiment.

    Parameters
    ----------
    name : str

    test : object

    stopping_rule : str (default="expected_loss")

    epsilon : float (default=1e-5)

    min_samples : int or None (default=None)
        The minimum number of samples for any variant.

    max_samples : int or None (default=None)
        The maximum number of samples for any variant.

    verbose : int or bool (default=False)
        Controls verbosity of output.

    Attributes
    ----------
    variants_ : list
        The variants in the experiment.

    n_variants_ : int
        The number of variants in the experiment.

    n_samples_ : int
        The total number of samples of all variants throughout the
        experimentation.

    n_updates_ : int
        The total number of updates throughout the experimentation.
    """
    def __init__(self, name, test, stopping_rule="expected_loss", epsilon=1e-5,
                 min_samples=None, max_samples=None, verbose=False):

        self.name = name
        self.test = test
        self.stopping_rule = stopping_rule
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.verbose = verbose
        # options

        # attributes
        self.variants_ = None
        self.n_variants_ = None
        self.n_samples_ = None
        self.n_updates_ = None

        # auxiliary data
        self._test = None
        self._test_type = None

        # statistics
        self._trials = {}

        # running statistics (mean / variance)

        # timing
        self._time_init = None
        self._time_termination = None

        # flags
        self._status = None
        self._termination = False

        self._setup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def run_update(self, **kwargs):
        """
        Parameters
        ----------
        """
        if self._termination:
            print("Experiment is terminated.")
            return

        self._update_data(**kwargs)

        self._update_stats(**kwargs)

        self._compute_metric()

        self._check_termination()

    def stats(self):
        """
        """
        pass

    def summary(self):
        """
        """
        pass

    def _check_termination(self):
        """"""
        variants = list(self._test.models.keys())

        if self.stopping_rule in ("expected_loss", "probability"):
            variants.remove("A")
        
        variant_metrics = sorted([(v, self._trials[v]["metric"][-1])
                                  for v in variants], key=lambda tup: tup[1])

        largest_metric = variant_metrics[-1]
        smallest_metric = variant_metrics[0]

        winner = ""
        if self.stopping_rule in ("expected_loss", "expected_loss_vs_all"):
            if smallest_metric[1] < self.epsilon:
                winner = smallest_metric[0]
        elif self.stopping_rule in ("probability", "probability_vs_all"):
            if largest_metric[1] > self.epsilon:
                winner = largest_metric[0]

        if self.min_samples is not None or self.max_samples is not None:
            variant_samples = [self._test.models[v].n_samples_
                               for v in self.variants_]

            min_samples = np.min(variant_samples)
            max_samples = np.max(variant_samples)

            min_stop_criterion = (self.min_samples is not None and
                                  min_samples >= self.min_samples)
            max_stop_criterion = (self.max_samples is not None and
                                  max_samples >= self.max_samples)

            if winner and min_stop_criterion:
                self._status = _STATUS_WINNER.format(winner)
                self._termination = True
            elif max_stop_criterion:
                self._status = _STATUS_MAX_SAMPLES
                self._termination = True
            else:
                self._status = _STATUS_RUNNING
        elif winner:
            self._status = _STATUS_WINNER.format(winner)
            self._termination = True
        else:
            self._status = _STATUS_RUNNING

    def _compute_metric(self):
        """"""
        variants = list(self._test.models.keys())

        if self.stopping_rule in ("expected_loss", "probability"):
            variants.remove("A")
            for variant in variants:
                if self.stopping_rule == "expected_loss":
                    _metric = self._test.expected_loss(variant=variant)
                else:
                    _metric = self._test.probability(variant=variant)

                self._trials[variant]["metric"].append(_metric)

        elif self.stopping_rule in ("expected_loss_vs_all",
                                    "probability_vs_all"):
            for variant in variants:
                if self.stopping_rule == "expected_loss_vs_all":
                    if self._test_type == "abtest":
                        control = next(v for v in variants if v != variant)
                        _metric = self._test.expected_loss(control=control,
                            variant=variant)
                    else:
                        _metric = self._test.expected_loss_vs_all(
                            variant=variant)
                else:
                    if self._test_type == "abtest":
                        control = next(v for v in variants if v != variant)
                        _metric = self._test.probability(control=control,
                            variant=variant)            
                    else:
                        _metric = self._test.probability_vs_all(
                            variant=variant)

                self._trials[variant]["metric"].append(_metric)

    def _setup(self):
        """"""
        if self.stopping_rule not in _STOPPING_RULES:
            raise ValueError("Stopping rule '{}' is not valid. "
                             "Available methods are {}"
                             .format(self.stopping_rule, _STOPPING_RULES))

        if self.min_samples is not None:
            if (not isinstance(self.min_samples, numbers.Number) or
                    self.min_samples < 0):
                raise ValueError("Minimum number of samples must be positive; "
                    "got {}.".format(self.min_samples))

        if self.max_samples is not None:
            if (not isinstance(self.max_samples, numbers.Number) or
                    self.max_samples < 0):
                raise ValueError("Maximum number of samples must be positive; "
                    "got {}.".format(self.min_samples))

        if None not in (self.min_samples, self.max_samples):
            if self.min_samples > self.max_samples:
                raise ValueError("min_samples must be <= max_samples.")

        if not isinstance(self.test, BayesMVTest):
            raise TypeError("test is not an instance inherited from "
                            "BayesMVTest.")

        self._test = copy.deepcopy(self.test)

        self.variants_ = self._test.models.keys()
        self.n_variants_ = len(self.variants_)

        if self.n_variants_ == 2:
            self._test_type = "abtest"
        else:
            self._test_type = "mvtest"
        
        for variant in self._test.models.keys():
            self._trials[variant] = {
                "datetime": [],
                "metric": [],
                "stats": {"mean": [], "var": [], "min": [], "max": []}
            }

        self._status = _STATUS_RUNNING

    def _update_data(self, **kwargs):
        """"""
        for variant, data in kwargs.items():
            self._test.update(variant=variant, data=data)

    def _update_stats(self, **kwargs):
        """"""
        pass

    @property
    def status(self):
        return self._status

    @property
    def termination(self):
        return self._termination
