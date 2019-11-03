"""
Bayesian experiment testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import pandas as pd

from pytest import raises
from scipy import stats

from cprior.experiment import Experiment
from cprior.experiment import experiment_describe
from cprior.experiment import experiment_plot_metric
from cprior.experiment import experiment_plot_stats
from cprior.experiment import experiment_stats
from cprior.experiment import experiment_summary
from cprior.models import BernoulliModel
from cprior.models import BernoulliMVTest
from cprior.models import NormalModel
from cprior.models import NormalMVTest


def test_experiment_input():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    with raises(ValueError):
        Experiment(name="CTR", test=test, stopping_rule="new_stopping_rule")

    with raises(ValueError):
        Experiment(name="CTR", test=test, min_n_samples=-10)

    with raises(ValueError):
        Experiment(name="CTR", test=test, max_n_samples=-10)

    with raises(ValueError):
        Experiment(name="CTR", test=test, min_n_samples=100, max_n_samples=10)

    with raises(TypeError):
        Experiment(name="CTR", test=None)


def test_experiment_bernoulli_probability():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test, stopping_rule="probability",
                            epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B})

    assert experiment.termination
    assert experiment.winner == "B"


def test_experiment_bernoulli_one_update():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test, stopping_rule="probability",
                            epsilon=0.99)

    with experiment as e:
        while not e.termination:
            data_A = stats.bernoulli(p=0.0223).rvs()
            data_B = stats.bernoulli(p=0.1128).rvs()

            e.run_update(**{"A": data_A, "B": data_B})

        assert experiment.termination
        assert experiment.winner == "B"


def test_experiment_bernoulli_expected_loss():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test,
                            stopping_rule="expected_loss", epsilon=1e-5)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B})

    assert experiment.termination
    assert experiment.winner == "B"


def test_experiment_bernoulli_probability_vs_all():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    modelC = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB, "C": modelC})

    experiment = Experiment(name="CTR", test=test,
                            stopping_rule="probability_vs_all", epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)
    data_C = stats.bernoulli(p=0.0528).rvs(size=900, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B, "C": data_C})

    assert experiment.termination
    assert experiment.winner == "B"


def test_experiment_bernoulli_probability_vs_all_ab():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test,
                            stopping_rule="probability_vs_all", epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B})

    assert experiment.termination
    assert experiment.winner == "B"


def test_experiment_bernoulli_expected_loss_vs_all():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    modelC = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB, "C": modelC})

    experiment = Experiment(name="CTR", test=test,
                            stopping_rule="expected_loss_vs_all", epsilon=1e-5)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)
    data_C = stats.bernoulli(p=0.0528).rvs(size=900, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B, "C": data_C})

    assert experiment.termination
    assert experiment.winner == "B"


def test_experiment_bernoulli_expected_loss_vs_all_ab():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test,
                            stopping_rule="expected_loss_vs_all", epsilon=1e-5)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B})

    assert experiment.termination
    assert experiment.winner == "B"


def test_experiment_bernoulli_probability_min_samples():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)

    abtest = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=abtest,
                            stopping_rule="probability", epsilon=0.99,
                            min_n_samples=1000)

    with experiment as e:
        data_A = stats.bernoulli(p=0.0223).rvs(size=90, random_state=42)
        data_B = stats.bernoulli(p=0.1128).rvs(size=50, random_state=42)

        e.run_update(**{"A": data_A, "B": data_B})

        assert e.termination is False


def test_experiment_bernoulli_probability_min_samples_2():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)

    abtest = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=abtest,
                            stopping_rule="probability", epsilon=0.99,
                            min_n_samples=100)

    with experiment as e:
        data_A = stats.bernoulli(p=0.0223).rvs(size=190, random_state=42)
        data_B = stats.bernoulli(p=0.1128).rvs(size=150, random_state=42)

        e.run_update(**{"A": data_A, "B": data_B})

        assert e.termination is True
        assert e.winner == "B"


def test_experiment_bernoulli_probability_max_samples():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)

    abtest = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=abtest,
                            stopping_rule="probability", epsilon=0.99,
                            max_n_samples=100)

    with experiment as e:
        data_A = stats.bernoulli(p=0.1123).rvs(size=100, random_state=42)
        data_B = stats.bernoulli(p=0.1128).rvs(size=100, random_state=42)

        e.run_update(**{"A": data_A, "B": data_B})

        assert e.termination is True
        assert e.winner is None


def test_experiment_save():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)

    abtest = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=abtest,
                            stopping_rule="probability", epsilon=0.99,
                            min_n_samples=100)

    data_A = stats.bernoulli(p=0.0223).rvs(size=190, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=150, random_state=42)

    experiment.run_update(**{"A": data_A, "B": data_B})
    experiment.run_update(**{"A": data_A, "B": data_B})

    with raises(TypeError):
        experiment.save(None)

    experiment.save("experiment_ctr.pkl")


def test_experiment_load():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)

    abtest = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=abtest,
                            stopping_rule="probability", epsilon=0.99,
                            min_n_samples=100)

    with raises(TypeError):
        experiment.load(None)

    experiment.load("experiment_ctr.pkl")

    assert experiment.termination is True
    assert experiment.winner == "B"
    assert experiment.status == "winner B"


def test_experiment_normal_mu():
    modelA = NormalModel(name="control")
    modelB = NormalModel(name="variation")
    modelC = NormalModel(name="variation")
    modelD = NormalModel(name="variation")

    mvtest = NormalMVTest({"A": modelA, "B": modelB, "C": modelC, "D": modelD})

    experiment = Experiment(name="GPA", test=mvtest,
                            stopping_rule="probability_vs_all", epsilon=0.99,
                            min_n_samples=200, max_n_samples=1000,
                            nig_metric="mu")

    with experiment as e:
        while not e.termination:
            data_A = stats.norm(loc=8, scale=3).rvs(size=10, random_state=42)
            data_B = stats.norm(loc=7, scale=2).rvs(size=25, random_state=42)
            data_C = stats.norm(loc=7.5, scale=4).rvs(size=12, random_state=42)
            data_D = stats.norm(loc=6.75, scale=2).rvs(size=8, random_state=42)

            e.run_update(**{"A": data_A, "B": data_B, "C": data_C,
                            "D": data_D})

        assert e.termination is True


def test_experiment_normal_sigma_sq():
    modelA = NormalModel(name="control")
    modelB = NormalModel(name="variation")
    modelC = NormalModel(name="variation")
    modelD = NormalModel(name="variation")

    mvtest = NormalMVTest({"A": modelA, "B": modelB, "C": modelC, "D": modelD})

    experiment = Experiment(name="GPA", test=mvtest,
                            stopping_rule="expected_loss_vs_all", epsilon=0.99,
                            min_n_samples=200, max_n_samples=1000,
                            nig_metric="sigma_sq")

    with experiment as e:
        while not e.termination:
            data_A = stats.norm(loc=8, scale=3).rvs(size=10, random_state=42)
            data_B = stats.norm(loc=7, scale=2).rvs(size=25, random_state=42)
            data_C = stats.norm(loc=7.5, scale=4).rvs(size=12, random_state=42)
            data_D = stats.norm(loc=6.75, scale=2).rvs(size=8, random_state=42)

            e.run_update(**{"A": data_A, "B": data_B, "C": data_C,
                            "D": data_D})

        assert e.termination is True


def test_experiment_normal_default():
    modelA = NormalModel(name="control")
    modelB = NormalModel(name="variation")
    modelC = NormalModel(name="variation")
    modelD = NormalModel(name="variation")

    mvtest = NormalMVTest({"A": modelA, "B": modelB, "C": modelC, "D": modelD})

    with raises(ValueError):
        experiment = Experiment(name="GPA", test=mvtest,
                                stopping_rule="probability_vs_all",
                                epsilon=0.99, min_n_samples=200,
                                max_n_samples=1000, nig_metric="new")

    experiment = Experiment(name="GPA", test=mvtest,
                            stopping_rule="probability_vs_all", epsilon=0.99,
                            min_n_samples=200, max_n_samples=1000)

    with experiment as e:
        while not e.termination:
            data_A = stats.norm(loc=8, scale=3).rvs(size=10, random_state=42)
            data_B = stats.norm(loc=7, scale=2).rvs(size=25, random_state=42)
            data_C = stats.norm(loc=7.5, scale=4).rvs(size=12, random_state=42)
            data_D = stats.norm(loc=6.75, scale=2).rvs(size=8, random_state=42)

            e.run_update(**{"A": data_A, "B": data_B, "C": data_C,
                            "D": data_D})

        assert e.termination is True


def test_experiment_stats():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test, stopping_rule="probability",
                            epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)
    experiment.run_update(**{"A": data_A, "B": data_B})

    with raises(TypeError):
        experiment_stats(None)

    assert isinstance(experiment_stats(experiment), pd.DataFrame)


def test_experiment_describe():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test, stopping_rule="probability",
                            epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)
    experiment.run_update(**{"A": data_A, "B": data_B})

    with raises(TypeError):
        experiment_describe(None)

    experiment.describe()


def test_experiment_summary():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test, stopping_rule="probability",
                            epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)
    experiment.run_update(**{"A": data_A, "B": data_B})

    with raises(TypeError):
        experiment_summary(None)

    experiment.summary()


def test_experiment_plots():
    modelA = BernoulliModel(name="control", alpha=1, beta=1)
    modelB = BernoulliModel(name="variation", alpha=1, beta=1)
    test = BernoulliMVTest({"A": modelA, "B": modelB})

    experiment = Experiment(name="CTR", test=test, stopping_rule="probability",
                            epsilon=0.99)

    data_A = stats.bernoulli(p=0.0223).rvs(size=1500, random_state=42)
    data_B = stats.bernoulli(p=0.1128).rvs(size=1100, random_state=42)
    experiment.run_update(**{"A": data_A, "B": data_B})

    with raises(TypeError):
        experiment_plot_metric(None)

    with raises(TypeError):
        experiment_plot_stats(None)

    experiment.plot_metric()
    experiment.plot_stats()


def test_experiment_normal_describe_plots():
    modelA = NormalModel(name="control")
    modelB = NormalModel(name="variation")
    modelC = NormalModel(name="variation")
    modelD = NormalModel(name="variation")

    mvtest = NormalMVTest({"A": modelA, "B": modelB, "C": modelC, "D": modelD})

    experiment = Experiment(name="GPA", test=mvtest,
                            stopping_rule="probability_vs_all", epsilon=0.99,
                            min_n_samples=200, max_n_samples=1000)

    with experiment as e:
        while not e.termination:
            data_A = stats.norm(loc=8, scale=3).rvs(size=10, random_state=42)
            data_B = stats.norm(loc=7, scale=2).rvs(size=25, random_state=42)
            data_C = stats.norm(loc=7.5, scale=4).rvs(size=12, random_state=42)
            data_D = stats.norm(loc=6.75, scale=2).rvs(size=8, random_state=42)

            e.run_update(**{"A": data_A, "B": data_B, "C": data_C,
                            "D": data_D})

    experiment.describe()
    experiment.summary()
    experiment.plot_metric()
