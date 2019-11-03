"""
Experiment utilities functions.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import pandas as pd

from ..cdist import BetaModel
from ..cdist import GammaModel
from ..cdist import NormalInverseGammaModel
from ..cdist import ParetoModel
from .base import Experiment


def experiment_stats(experiment):
    """
    Experiment main statistics on collected data.

    Parameters
    ----------
    experiment : object

    Returns
    -------
    stats : pandas.DataFrame
    """
    if not isinstance(experiment, Experiment):
        raise TypeError()

    d_cols = []
    for variant in experiment.variants_:
        d_cols.append(pd.Series(experiment._trials[variant]["data"],
                      name=variant).describe())

    return pd.concat(d_cols, axis=1)


def experiment_describe(experiment):
    """
    Experiment settings.

    Parameters
    ----------
    experiment : object
    """
    if not isinstance(experiment, Experiment):
        raise TypeError()

    # Experiment class arguments
    name = experiment.name
    stopping_rule = experiment.stopping_rule
    epsilon = experiment.epsilon
    min_n_samples = experiment.min_n_samples
    max_n_samples = experiment.max_n_samples

    min_n_samples = "not set" if min_n_samples is None else min_n_samples
    max_n_samples = "not set" if max_n_samples is None else max_n_samples

    # Experiment class attributes
    variants = experiment.variants_
    n_variants = experiment.n_variants_

    # others
    models = experiment._test.models
    model_type = type(models["A"])
    model_type_name = model_type.__name__.lower()[:-5]

    if issubclass(model_type, BetaModel):
        model_prior_type_name = "beta"

        prior_params = ["alpha", "beta"]
        prior_values = [(models[v].alpha, models[v].beta) for v in variants]
    elif issubclass(model_type, GammaModel):
        model_prior_type_name = "gamma"

        prior_params = ["shape", "rate"]
        prior_values = [(models[v].shape, models[v].rate) for v in variants]
    elif issubclass(model_type, NormalInverseGammaModel):
        model_prior_type_name = "normalinversegamma"

        prior_params = ["loc", "variance_scale", "shape", "scale"]
        prior_values = [(models[v].loc, models[v].variance_scale,
                         models[v].shape, models[v].scale) for v in variants]
    elif issubclass(model_type, ParetoModel):
        model_prior_type_name = "pareto"

        prior_params = ["scale", "shape"]
        prior_values = [(models[v].scale, models[v].shape) for v in variants]

    bayesian_model = "{}-{}".format(model_type_name, model_prior_type_name)

    df_priors = pd.DataFrame(prior_values, columns=prior_params)
    df_priors = df_priors.rename(index=dict(zip(range(n_variants), variants)))
    df_priors_string = " "*6 + df_priors.to_string().replace("\n", "\n      ")

    report = (
        "=====================================================\n"
        "  Experiment: {}\n"
        "=====================================================\n"
        "    Bayesian model:     {:>25}\n"
        "    Number of variants: {:>25}\n"
        "\n"
        "    Options:\n"
        "      stopping rule     {:>25}\n"
        "      epsilon           {:>25.5f}\n"
        "      min_n_samples     {:>25}\n"
        "      max_n_samples     {:>25}\n"
        "\n"
        "    Priors:\n\n{}\n"
        "  -------------------------------------------------\n"
        ).format(name, bayesian_model, n_variants, stopping_rule, epsilon,
                 min_n_samples, max_n_samples, df_priors_string)

    print(report)


def experiment_summary(experiment):
    """
    Experiment summary with several decision metrics.

    If a winner has been declared, the corresponding row is highlighted
    in green.

    Parameters
    ----------
    experiment : object

    Returns
    -------
    stats : pandas.DataFrame
    """
    if not isinstance(experiment, Experiment):
        raise TypeError()

    test = experiment._test
    winner = experiment.winner

    if experiment._multimetric:
        multi_idx = experiment._multimetric_idx

    report = {}

    for variant in experiment.variants_:
        if variant != "A":
            # compute probability and expected_loss
            probability = test.probability(variant=variant)
            expected_loss = test.expected_loss(variant=variant)
            expected_loss_rel = test.expected_loss_relative(variant=variant)

            if experiment._multimetric:
                probability = probability[multi_idx]
                expected_loss = expected_loss[multi_idx]
                improvement = -expected_loss_rel[multi_idx]
            else:
                improvement = -expected_loss_rel

            probability_str = "{:.2%}".format(probability)
            improvement_str = "{:.2%}".format(improvement)
        else:
            probability_str = "-"
            expected_loss = "-"
            improvement_str = "-"

        probability_vs_all = test.probability_vs_all(variant=variant)
        expected_loss_vs_all = test.expected_loss_vs_all(variant=variant)
        expected_loss_rel_vs_all = test.expected_loss_relative_vs_all(
            variant=variant)

        if experiment._multimetric:
            probability_vs_all = probability_vs_all[multi_idx]
            expected_loss_vs_all = expected_loss_vs_all[multi_idx]
            improvement_vs_all = -expected_loss_rel_vs_all[multi_idx]
        else:
            improvement_vs_all = -expected_loss_rel_vs_all

        probability_vs_all_str = "{:.2%}".format(probability_vs_all)
        improvement_vs_all_str = "{:.2%}".format(improvement_vs_all)

        report[variant] = {
            "name": test.models[variant].name,
            "probability": probability_str,
            "expected_loss": expected_loss,
            "probability_vs_all": probability_vs_all_str,
            "expected_loss_vs_all": expected_loss_vs_all,
            "improvement": improvement_str,
            "improvement_vs_all": improvement_vs_all_str,
            "n_samples": test.models[variant].n_samples_
        }

    cols = ["name", "probability", "expected_loss", "improvement",
            "probability_vs_all", "expected_loss_vs_all",
            "improvement_vs_all", "n_samples"]

    df_report = pd.DataFrame.from_dict(report).T
    df_report = df_report[cols]

    if winner is not None:
        return df_report.style.set_properties(
            subset=pd.IndexSlice[[winner], :],
            **{'background-color': "#C4F4C5", 'font-weight': 'bold'})
    else:
        return df_report
