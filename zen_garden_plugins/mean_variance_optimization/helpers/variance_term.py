"""Build the variance of the net present cost as a quadratic linopy expression.

For every variance source (see ``variance_sources.py``) the plugin adds:

1. one auxiliary variable per uncertain element ``e``::

       y_e = Σ_{i of e} σ_i · x_i

2. the quadratic term::

       Var = Σ_e Σ_e' R[e, e'] · y_e · y_e'

The auxiliary variables keep the quadratic term small: it has one entry per
pair of elements instead of one per pair of model entries.
"""

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from zen_garden_plugins.mean_variance_optimization.helpers.variance_sources import (
    VarianceSource,
)


def add_weighted_sd_sum(optimization_model, source: VarianceSource) -> linopy.Variable:
    """Add the variable ``y_e = Σ_{i of e} σ_i · x_i`` and its defining constraint."""
    lp_model = optimization_model.lp_model
    name = source.weighted_sum_variable
    elements = pd.Index(source.uncertain_elements, name=source.element_dimension)

    lp_model.add_variables(name=name, coords=[elements])
    # register the variable so that ZEN-garden saves it with the results
    optimization_model.variables.docs[name] = (
        f"Sum of {source.decision_variable} weighted with the standard deviation "
        f"of its discounted cost, per {source.element_dimension} "
        "(mean_variance_optimization plugin)"
    )
    optimization_model.variables.units[name] = None
    weighted_sd_sum = lp_model.variables[name]

    # Decision variable is capacity_addition or import
    decision_variable = optimization_model.variables[source.decision_variable]
    decision_variable = decision_variable.sel({source.element_dimension: elements})
    sum_of_sd_times_x = (source.absolute_sd * decision_variable).sum(
        source.summed_dimensions
    )
    optimization_model.add_constraint(
        f"constraint_{name}", weighted_sd_sum == sum_of_sd_times_x
    )
    return weighted_sd_sum


def quadratic_variance(
    weighted_sd_sum: linopy.Variable, source: VarianceSource, include_correlation: bool
) -> linopy.QuadraticExpression:
    """Return ``Var = Σ_e Σ_e' R[e, e'] · y_e · y_e'``.

    Without correlation, only the diagonal (``R[e, e] = 1``) is kept.
    """
    correlation = _correlation_used(source, include_correlation)
    dimension = source.element_dimension

    y_first = weighted_sd_sum.rename({dimension: "element_1"})
    y_second = weighted_sd_sum.rename({dimension: "element_2"})
    correlation_array = xr.DataArray(
        correlation.values,
        coords={"element_1": correlation.index, "element_2": correlation.columns},
        dims=["element_1", "element_2"],
    )
    return (correlation_array * y_first * y_second).sum()


def variance_from_solution(
    optimization_model, source: VarianceSource, include_correlation: bool
) -> pd.DataFrame:
    """Evaluate the variance for the solved model, per pair of elements.

    Computed from the solution of the decision variable, so it also works when
    the variance was not part of the objective (weighting factor 0).

    Returns:
        One row per pair of elements with the columns ``element_1``,
        ``element_2``, ``correlation``, ``weighted_sd_sum_1``,
        ``weighted_sd_sum_2`` and ``variance_contribution``. The sum of
        ``variance_contribution`` is the variance of this source.
    """
    decision_variable = optimization_model.variables[source.decision_variable]
    solution = decision_variable.solution.sel(
        {source.element_dimension: source.uncertain_elements}
    ).fillna(0.0)
    weighted_sd_sum = (
        (source.absolute_sd * solution).sum(source.summed_dimensions).to_series()
    )
    weighted_sd_sum = weighted_sd_sum.loc[source.uncertain_elements]

    correlation = _correlation_used(source, include_correlation)
    contributions = correlation * np.outer(weighted_sd_sum, weighted_sd_sum)

    table = correlation.stack().rename("correlation").to_frame()
    table.index.names = ["element_1", "element_2"]
    table["weighted_sd_sum_1"] = weighted_sd_sum.loc[
        table.index.get_level_values("element_1")
    ].values
    table["weighted_sd_sum_2"] = weighted_sd_sum.loc[
        table.index.get_level_values("element_2")
    ].values
    table["variance_contribution"] = contributions.stack().values
    return table.reset_index()


def _correlation_used(
    source: VarianceSource, include_correlation: bool
) -> pd.DataFrame:
    """Return the correlation matrix, or only its diagonal without correlation."""
    if include_correlation:
        return source.correlation
    diagonal_only = np.diag(np.diag(source.correlation.values))
    return pd.DataFrame(
        diagonal_only,
        index=source.correlation.index,
        columns=source.correlation.columns,
    )
