"""Mean-variance optimization plugin for ZEN-garden.

The plugin adds the variance of the net present cost (NPC) to the optimization.
The variance stems from uncertain technology capex and carrier import prices.
Three methods are available (``method`` in the config):

- ``weighting_factor``: minimize ``NPC + weighting_factor · Var``
- ``cost_constraint``: minimize ``Var`` subject to ``NPC <= cost_budget``
- ``regularization_only``: minimize ``NPC + regularization_factor · Σ x²``
  over all variables (no variance; makes the solution unique)

How the variance is built is explained in ``variance_sources.py`` and
``variance_term.py``. The input files are described in ``input_data.py``.

The plugin uses two ZEN-garden events:

- ``after_model_construction``: replaces the objective.
- ``after_postprocessing``: writes ``mean_variance.json`` and
  ``mean_variance_contributions.csv`` to the output folder.
"""

from typing import Literal

from pydantic import model_validator
from zen_garden import ConfigBase, Event, EventPublisher

from zen_garden_plugins.mean_variance_optimization.helpers.reporting import (
    write_mean_variance_report,
)
from zen_garden_plugins.mean_variance_optimization.helpers.variance_sources import (
    create_variance_sources,
)
from zen_garden_plugins.mean_variance_optimization.helpers.variance_term import (
    add_local_weighted_sd_sum,
    add_weighted_sd_sum,
    quadratic_variance,
)

PLUGIN_NAME = "mean_variance_optimization"
# Built optimization steps are kept in this list
BUILT_OPTIMIZATION_STEP = []

class Config(ConfigBase):
    """Settings under ``plugins.mean_variance_optimization`` in the config file.

    Example::

        plugins:
          mean_variance_optimization:
            method: weighting_factor
            weighting_factor: 1.0e-8
            include_variances_for: ["technology_capex", "imports"]
    """

    #: Objective, see the module docstring.
    method: Literal["weighting_factor", "cost_constraint", "regularization_only"] = (
        "weighting_factor"
    )
    #: Weight of the variance for ``method: weighting_factor`` (0 = only NPC).
    weighting_factor: float = 0.0
    #: Upper limit of the NPC for ``method: cost_constraint``.
    cost_budget: float | None = None
    #: Uncertain costs included in the variance.
    include_variances_for: list[Literal["technology_capex", "imports"]] = [
        "technology_capex",
        "imports",
    ]
    #: If false, only the variances (diagonal of the correlation matrix) are used.
    include_correlation: bool = True
    #: Shrinkage of the correlation matrix towards the identity, which keeps it
    #: numerically positive definite: ``R_used = (1 - λ) · R + λ · I``.
    correlation_regularization: float = 1e-6
    #: Weight of ``Σ x²`` for ``method: regularization_only``.
    regularization_factor: float | None = None

    @model_validator(mode="after")
    def _check_method_settings(self):
        """Make sure the settings needed by the selected method are given."""
        if self.method == "cost_constraint" and self.cost_budget is None:
            raise ValueError("method 'cost_constraint' requires a 'cost_budget'.")
        if self.method == "cost_constraint" and not self.include_variances_for:
            raise ValueError(
                "method 'cost_constraint' minimizes the variance, so "
                "'include_variances_for' must not be empty."
            )
        if self.method == "regularization_only" and self.regularization_factor is None:
            raise ValueError("method 'regularization_only' requires a"
                             " 'regularization_factor'.")
        return self


@EventPublisher.register(Event.after_model_construction)
def keep_built_model(optimization_step) -> None:
    BUILT_OPTIMIZATION_STEP.append(optimization_step)

@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_step) -> None:
    """Replace the objective of ZEN-garden according to the selected method.

    Args:
        optimization_step: The ZEN-garden optimization step whose model has
            just been constructed.
    """
    optimization_model = optimization_step.optimization_model
    _raise_if_scaling_is_used(optimization_model)

    time_steps = optimization_step.time_steps
    settings = optimization_model.config.plugins.get(PLUGIN_NAME)
    if settings is None:  # plugin installed, but not used in this run
        return

    method = settings["method"]
    lp_model = optimization_model.lp_model
    net_present_cost = optimization_model.expressions["total_cost"]

    if method == "weighting_factor" and settings["weighting_factor"] == 0:
        return  # the objective of ZEN-garden is already the NPC

    lp_model.remove_objective()

    if method == "weighting_factor":
        variance = _build_variance(optimization_model, time_steps, settings)

        lp_model.add_objective(
            net_present_cost + settings["weighting_factor"] * variance, sense="min"
        )

    elif method == "cost_constraint":
        variance = _build_variance(optimization_model, time_steps, settings)

        lp_model.add_objective(variance, sense="min")

        optimization_model.add_constraint(
            "constraint_mean_variance_cost_budget",
            net_present_cost <= settings["cost_budget"],
        )

    elif method == "regularization_only":
        lp_model.add_objective(
            net_present_cost
            + settings["regularization_factor"] * _sum_of_squares(lp_model),
            sense="min",
        )


@EventPublisher.register(Event.after_postprocessing)
def report_mean_variance(postprocessing) -> None:
    """Write the variance of the solution to the output folder."""
    settings = postprocessing.config.plugins.get(PLUGIN_NAME)
    if settings is None:  # plugin installed, but not used in this run
        return
    write_mean_variance_report(postprocessing, settings)


def _build_variance(optimization_model, time_steps, settings):
    """Return the variance of the NPC summed over all variance sources."""
    variance = 0
    variance_sources = create_variance_sources(optimization_model, time_steps, settings)
    for source in variance_sources:
        weighted_sd_sum = add_weighted_sd_sum(optimization_model, source)
        local_weighted_sd_sum = add_local_weighted_sd_sum(optimization_model, source)
        variance += quadratic_variance(
            weighted_sd_sum,
            local_weighted_sd_sum,
            source,
            settings["include_correlation"],
        )
    return variance


def _sum_of_squares(lp_model):
    """Return ``Σ x²`` over all variables of the model."""
    # Todo: add subset of variables to settings
    sum_of_squares = 0
    for name in lp_model.variables:
        variable = lp_model.variables[name]
        sum_of_squares += (variable * variable).sum()
    return sum_of_squares


def _raise_if_scaling_is_used(optimization_model) -> None:
    """The scaling of ZEN-garden only supports linear objectives."""
    if optimization_model.config.solver.use_scaling:
        raise ValueError(
            "The scaling of ZEN-garden does not support the quadratic objective of "
            "the mean-variance optimization. Set solver.use_scaling to false."
        )
