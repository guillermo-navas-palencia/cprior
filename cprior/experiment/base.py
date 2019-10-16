"""
Bayesian A/B and MV testing experiment.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import copy
import datetime
import numbers
import time

import numpy as np

from ..cdist.base import BayesMVTest
from .plotting import experiment_plot_metric
from .plotting import experiment_plot_stats
from .utils import experiment_describe
from .utils import experiment_stats
from .utils import experiment_summary


_STATUS_max_n_samples = "max_n_samples exceeded"
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

    min_n_samples : int or None (default=None)
        The minimum number of samples for any variant.

    max_n_samples : int or None (default=None)
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
                 min_n_samples=None, max_n_samples=None, verbose=False,
                 **options):

        self.name = name
        self.test = test
        self.stopping_rule = stopping_rule
        self.epsilon = epsilon
        self.min_n_samples = min_n_samples
        self.max_n_samples = max_n_samples
        self.verbose = verbose

        # options
        self._method = options.get("method", None)
        self._nig_metric = options.get("nig_metric", None)

        # attributes
        self.variants_ = None
        self.n_variants_ = None
        self.n_samples_ = None
        self.n_updates_ = None

        # auxiliary data
        self._test = None
        self._test_type = None
        self._multimetric = False
        self._multimetric_idx = None

        # statistics
        self._trials = {}

        # timing
        self._time_init = None
        self._time_termination = None

        # flags
        self._status = None
        self._termination = False
        self._winner = None

        self._setup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def describe(self):
        """"""
        return experiment_describe(self)

    def plot_metric(self):
        """"""
        return experiment_plot_metric(self)

    def plot_stats(self):
        """"""
        return experiment_plot_stats(self)

    def run_update(self, **kwargs):
        """
        Parameters
        ----------
        """
        if self._termination:
            print("Experiment is terminated.")
            return

        self._update_data(**kwargs)

        self._update_stats()

        self._compute_metric()

        self._check_termination()

    def stats(self):
        """"""
        return experiment_stats(self)

    def summary(self):
        """"""
        return experiment_summary(self)

    def _check_termination(self):
        """"""
        variants = list(self._test.models.keys())

        if self.stopping_rule in ("expected_loss", "probability"):
            variants.remove("A")
        
        if self._multimetric:
            variant_metrics = sorted(
                [(v, self._trials[v]["metric"][-1]) for v in variants],
                key=lambda tup: tup[1][self._multimetric_idx])
        else:
            variant_metrics = sorted([(v, self._trials[v]["metric"][-1])
                                      for v in variants],
                                      key=lambda tup: tup[1])

        largest_metric = variant_metrics[-1]
        smallest_metric = variant_metrics[0]

        winner = ""
        if self.stopping_rule in ("expected_loss", "expected_loss_vs_all"):
            variant, metric = smallest_metric

            if self._multimetric:
                metric = metric[self._multimetric_idx]

            if metric < self.epsilon:
                winner = variant
                self._winner = winner
        elif self.stopping_rule in ("probability", "probability_vs_all"):
            variant, metric = largest_metric

            if self._multimetric:
                metric = metric[self._multimetric_idx]

            if metric > self.epsilon:
                winner = variant
                self._winner = winner

        if self.min_n_samples is not None or self.max_n_samples is not None:
            variant_samples = [self._test.models[v].n_samples_
                               for v in self.variants_]

            min_n_samples = np.min(variant_samples)
            max_n_samples = np.max(variant_samples)

            min_stop_criterion = (self.min_n_samples is not None and
                                  min_n_samples >= self.min_n_samples)
            max_stop_criterion = (self.max_n_samples is not None and
                                  max_n_samples >= self.max_n_samples)

            if winner and min_stop_criterion:
                self._status = _STATUS_WINNER.format(winner)
                self._termination = True
            elif max_stop_criterion:
                self._status = _STATUS_max_n_samples
                self._termination = True
            else:
                self._status = _STATUS_RUNNING
        elif winner:
            self._status = _STATUS_WINNER.format(winner)
            self._termination = True
        else:
            self._status = _STATUS_RUNNING

        if self._termination:
            self._time_termination = time.perf_counter()

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
                        control = variants[not variants.index(variant)]
                        _metric = self._test.expected_loss(control=control,
                            variant=variant)
                    else:
                        _metric = self._test.expected_loss_vs_all(
                            variant=variant)
                else:
                    if self._test_type == "abtest":
                        control = variants[not variants.index(variant)]
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

        if self.min_n_samples is not None:
            if (not isinstance(self.min_n_samples, numbers.Number) or
                    self.min_n_samples < 0):
                raise ValueError("Minimum number of samples must be positive; "
                    "got {}.".format(self.min_n_samples))

        if self.max_n_samples is not None:
            if (not isinstance(self.max_n_samples, numbers.Number) or
                    self.max_n_samples < 0):
                raise ValueError("Maximum number of samples must be positive; "
                    "got {}.".format(self.min_n_samples))

        if None not in (self.min_n_samples, self.max_n_samples):
            if self.min_n_samples > self.max_n_samples:
                raise ValueError("min_n_samples must be <= max_n_samples.")

        if not isinstance(self.test, BayesMVTest):
            raise TypeError("test is not an instance inherited from "
                            "BayesMVTest.")

        self._time_init = time.perf_counter()

        # clone test to run experiment
        self._test = copy.deepcopy(self.test)

        self.variants_ = self._test.models.keys()
        self.n_variants_ = len(self.variants_)

        if self.n_variants_ == 2:
            self._test_type = "abtest"
        else:
            self._test_type = "mvtest"
        
        # initialize dictionary to store information of each variant at each
        # iteration/update.
        for variant in self._test.models.keys():
            self._trials[variant] = {
                "datetime": [],
                "metric": [],
                "data": [],
                "n_samples": [],
                "stats": {"mean": [], "ci_low": [], "ci_high": []}
            }

        # initialize status message
        self._status = _STATUS_RUNNING

        # extra options
        if type(self._test).__name__ in ("NormalMVTest", "LogNormalMVTest"):
            self._multimetric = True

            if self._nig_metric is not None:
                if self._nig_metric not in ("mu", "sigma_sq"):
                    raise ValueError()
                
                if self._nig_metric == "mu":
                    self._multimetric_idx = 0
                else:
                    self._multimetric_idx = 1
            else:
                self._nig_metric = "mu"  # default
                self._multimetric_idx = 0

    def _update_data(self, **kwargs):
        """"""
        update_datetime = str(datetime.datetime.now())

        for variant, data in kwargs.items():
            self._test.update(variant=variant, data=data)

            self._trials[variant]["datetime"].append(update_datetime)

            x = np.asarray(data)
            n = x.size

            if n > 1:
                self._trials[variant]["data"].extend(x)
            else:
                self._trials[variant]["data"].append(data)

            self._trials[variant]["n_samples"].append(n)

    def _update_stats(self):
        """"""
        for variant in self.variants_:
            mean = self._test.models[variant].mean()
            ci_low, ci_high = self._test.models[variant].ppf([0.05, 0.95])
            self._trials[variant]["stats"]["mean"].append(mean)
            self._trials[variant]["stats"]["ci_low"].append(ci_low)
            self._trials[variant]["stats"]["ci_high"].append(ci_high)

    @property
    def status(self):
        return self._status

    @property
    def termination(self):
        return self._termination

    @property
    def winner(self):
        return self._winner
