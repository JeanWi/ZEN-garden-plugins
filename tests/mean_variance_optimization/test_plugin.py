import json
import os
import shutil

import linopy as lp
import pytest
from zen_garden import ConfigBase, Event, EventPublisher, run

from zen_garden_plugins.mean_variance_optimization import plugin

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
DATASET = "test_mv"


@pytest.fixture(autouse=True)
def only_mean_variance_observers(monkeypatch):
    """Keep only the observers of this plugin registered during a test.

    Observers of other plugins imported by other tests stay registered in the same
    process and would otherwise be triggered in the runs of these tests.
    """
    observers = EventPublisher.observers()
    monkeypatch.setattr(
        EventPublisher,
        "_EventPublisher__observers",
        {
            event: [obs for obs in observers_event if obs.__module__ == plugin.__name__]
            for event, observers_event in observers.items()
        },
    )


def _run_with_plugin(tmp_path, plugin_settings, use_scaling=False):
    """Run the test dataset with the mean-variance plugin and return the workflow
    and the reported mean-variance results."""
    dataset_path = tmp_path / DATASET
    shutil.copytree(os.path.join(FIXTURES_PATH, DATASET), dataset_path)
    config = {
        "analysis": {"folder_output": str(tmp_path / "outputs")},
        "solver": {"keep_files": False, "use_scaling": use_scaling},
        "plugins": {"mean_variance_optimization": plugin_settings},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    workflow = run(config=config_path, dataset=dataset_path)

    report_path = tmp_path / "outputs" / DATASET / "mean_variance_dict.json"
    with open(report_path) as f:
        report = json.load(f)
    return workflow, report


def test_config_defaults():
    """Test the plugin config schema and its default values."""
    config = plugin.Config()

    assert issubclass(plugin.Config, ConfigBase)
    assert config.method == "weighting_factor"
    assert config.weighting_factor is None
    assert config.include_correlation
    assert config.include_variances_for == ["technology_capex", "imports"]


def test_config_rejects_unknown_variance_type():
    """Test that a misspelled variance type is not silently ignored."""
    with pytest.raises(ValueError):
        plugin.Config(include_variances_for=["import"])


def test_handlers_are_registered():
    """Test that the handlers are registered to the respective events."""
    observers = EventPublisher.observers()

    assert plugin.construct_mean_variance_objective in observers[
        Event.after_model_construction
    ]
    assert plugin.calculate_variance_from_solution in observers[
        Event.after_postprocessing
    ]


def test_mean_variance_objective(tmp_path):
    """Test that the objective is NPC + weighting_factor * variance of the solution."""
    weighting_factor = 1e-3
    workflow, report = _run_with_plugin(
        tmp_path,
        {
            "weighting_factor": weighting_factor,
            "include_variances_for": ["technology_capex", "imports"],
        },
    )
    lp_model = workflow.service_container.get("optimization_model").lp_model

    assert isinstance(lp_model.objective.expression, lp.QuadraticExpression)
    assert report["variance"] > 0
    assert report["weighting_factor"] == weighting_factor
    assert report["objective_value"] == pytest.approx(
        report["npv"] + weighting_factor * report["variance"], rel=1e-4
    )


def test_quadratic_objective_with_scaling_raises(tmp_path):
    """Test that the unsupported combination with scaling fails early."""
    with pytest.raises(ValueError, match="use_scaling"):
        _run_with_plugin(tmp_path, {"weighting_factor": 1e-3}, use_scaling=True)


def test_zero_weighting_factor_minimizes_npc(tmp_path):
    """Test that without weighting factor, the objective is the net present cost."""
    workflow, report = _run_with_plugin(tmp_path, {"weighting_factor": 0})
    lp_model = workflow.service_container.get("optimization_model").lp_model

    assert isinstance(lp_model.objective.expression, lp.LinearExpression)
    assert report["objective_value"] == pytest.approx(report["npv"], rel=1e-6)
