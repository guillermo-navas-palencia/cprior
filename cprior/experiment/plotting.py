"""
Experiment plotting functions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import matplotlib.pyplot as plt
import numpy as np


def experiment_plot_metric(experiment):
    """
    Parameters
    ----------
    experiment : object
    """
    for variant in experiment.variants_:
        if experiment._multimetric:
            metrics = list(zip(*experiment._trials[variant]["metric"]))
            try:
                metric = metrics[experiment._multimetric_idx]
            except:
                metric = []
        else:
            metric = experiment._trials[variant]["metric"]

        plt.plot(metric,label="Model {} ({})".format(
            variant,experiment._test.models[variant].name))

    plt.axhline(experiment.epsilon, linestyle="--", color="r")

    plt.title(experiment.stopping_rule)
    plt.xlabel("n_updates")
    plt.legend()
    plt.grid(True, color="grey", alpha=0.3)
    plt.show()


def experiment_plot_stats(experiment):
    """
    Parameters
    ----------
    experiment : object
    """
    for variant in experiment.variants_:
        if experiment._multimetric:
            pass
        else:
            mean = experiment._trials[variant]["stats"]["mean"]
            ci_low = experiment._trials[variant]["stats"]["ci_low"]
            ci_high = experiment._trials[variant]["stats"]["ci_high"]

            plt.plot(mean,label="Model {} ({})".format(
                variant,experiment._test.models[variant].name))
            plt.fill_between(np.arange(len(mean)), ci_low, ci_high, alpha=0.2)

    plt.title("mean and CI over time")
    plt.xlabel("n_updates")
    plt.ylabel("mean")
    plt.legend()
    plt.grid(True, color="grey", alpha=0.3)
    plt.show()
