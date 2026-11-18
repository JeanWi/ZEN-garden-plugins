"""Build the variance of the net present cost as a quadratic linopy expression.

For every variance source (see ``variance_sources.py``) the plugin adds:

1. one auxiliary variable per uncertain element ``e``::

       y_e = Σ_{i of e} σ_i · x_i

2. if some elements have a spatial correlation ``ρ_s[e] < 1``, one auxiliary
   variable per such element and location ``l``::

       z_{e,l} = Σ_{i of e at l} σ_i · x_i

3. the quadratic term::

       Var = Σ_e Σ_e' C[e, e'] · y_e · y_e' + Σ_l Σ_e Σ_e' L[e, e'] · z_{e,l} · z_{e',l}

   with the common part ``C[e, e'] = R[e, e'] · √(ρ_s[e] ρ_s[e'])`` and the
   local part ``L[e, e'] = R[e, e'] · √((1 - ρ_s[e])(1 - ρ_s[e']))``.

The auxiliary variables keep the quadratic term small: it has one entry per
pair of elements (and location) instead of one per pair of model entries.
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


def add_local_weighted_sd_sum(
    optimization_model, source: VarianceSource
) -> linopy.Variable | None:
    """Add the variable ``z_{e,l} = Σ_{i of e at l} σ_i · x_i`` and its constraint.

    Only added for the elements with a spatial correlation below 1, and only
    for the locations at which the element has a nonzero standard deviation
    (e.g. not for conversion technologies on edges).

    Returns:
        The variable, or ``None`` if all elements are perfectly correlated in
        space.
    """
    local_elements = source.local_elements
    if not local_elements:
        return None
    lp_model = optimization_model.lp_model
    name = source.local_weighted_sum_variable
    location_dimension = source.location_dimension

    absolute_sd = source.absolute_sd.sel({source.element_dimension: local_elements})
    is_uncertain = (absolute_sd != 0).any(source.summed_dimensions_per_location)
    is_uncertain = is_uncertain.transpose(source.element_dimension, location_dimension)

    lp_model.add_variables(
        name=name,
        coords=[
            pd.Index(local_elements, name=source.element_dimension),
            absolute_sd.indexes[location_dimension],
        ],
        mask=is_uncertain,
    )
    optimization_model.variables.docs[name] = (
        f"Sum of {source.decision_variable} weighted with the standard deviation "
        f"of its discounted cost, per {source.element_dimension} and "
        f"{location_dimension} (mean_variance_optimization plugin)"
    )
    optimization_model.variables.units[name] = None
    local_weighted_sd_sum = lp_model.variables[name]

    decision_variable = optimization_model.variables[source.decision_variable]
    decision_variable = decision_variable.sel({source.element_dimension: local_elements})
    sum_of_sd_times_x = (absolute_sd * decision_variable).sum(
        source.summed_dimensions_per_location
    )
    optimization_model.add_constraint(
        f"constraint_{name}",
        local_weighted_sd_sum == sum_of_sd_times_x,
        mask=is_uncertain,
    )
    return local_weighted_sd_sum


def quadratic_variance(
    weighted_sd_sum: linopy.Variable,
    local_weighted_sd_sum: linopy.Variable | None,
    source: VarianceSource,
    include_correlation: bool,
) -> linopy.QuadraticExpression:
    """Return ``Var = Σ C[e, e'] · y_e · y_e' + Σ_l Σ L[e, e'] · z_{e,l} · z_{e',l}``.

    Without correlation, only the diagonal of ``R`` (``R[e, e] = 1``) is kept.
    """
    variance = _pairwise_product(
        _common_correlation(source, include_correlation),
        weighted_sd_sum,
        source.element_dimension,
    )
    if local_weighted_sd_sum is not None:
        variance += _pairwise_product(
            _local_correlation(source, include_correlation),
            local_weighted_sd_sum,
            source.element_dimension,
        )
    return variance


def _pairwise_product(
    coefficients: pd.DataFrame, variable: linopy.Variable, dimension: str
) -> linopy.QuadraticExpression:
    """Return ``Σ_{e, e'} coefficients[e, e'] · v_e · v_e'`` (summed over all dims)."""
    v_first = variable.rename({dimension: "element_1"})
    v_second = variable.rename({dimension: "element_2"})
    coefficient_array = xr.DataArray(
        coefficients.values,
        coords={"element_1": coefficients.index, "element_2": coefficients.columns},
        dims=["element_1", "element_2"],
    )
    return (coefficient_array * v_first * v_second).sum()


def variance_from_solution(
    optimization_model, source: VarianceSource, include_correlation: bool
) -> pd.DataFrame:
    """Evaluate the variance for the solved model, per pair of elements.

    Computed from the solution of the decision variable, so it also works when
    the variance was not part of the objective (weighting factor 0).

    Returns:
        One row per pair of elements with the columns ``component``
        (``common`` or ``local``), ``location`` (only for ``local``),
        ``element_1``, ``element_2``, ``correlation`` (the coefficient ``C`` or
        ``L``), ``weighted_sd_sum_1``, ``weighted_sd_sum_2`` and
        ``variance_contribution``. Local rows without contribution are left
        out. The sum of ``variance_contribution`` is the variance of this
        source.
    """
    decision_variable = optimization_model.variables[source.decision_variable]
    solution = decision_variable.solution.sel(
        {source.element_dimension: source.uncertain_elements}
    ).fillna(0.0)
    sd_times_x = source.absolute_sd * solution

    weighted_sd_sum = sd_times_x.sum(source.summed_dimensions).to_series()
    weighted_sd_sum = weighted_sd_sum.loc[source.uncertain_elements]
    common = _pair_table(
        _common_correlation(source, include_correlation), weighted_sd_sum
    ).assign(component="common", location=None)
    tables = [common]

    if source.local_elements:
        location_dimension = source.location_dimension
        local_correlation = _local_correlation(source, include_correlation)
        local_weighted_sd_sum = (
            sd_times_x.sel({source.element_dimension: source.local_elements})
            .sum(source.summed_dimensions_per_location)
            .transpose(source.element_dimension, location_dimension)
            .to_pandas()
        )
        for location in local_weighted_sd_sum.columns:
            table = _pair_table(local_correlation, local_weighted_sd_sum[location])
            table = table[table["variance_contribution"] != 0]
            tables.append(table.assign(component="local", location=location))

    columns = [
        "component",
        "location",
        "element_1",
        "element_2",
        "correlation",
        "weighted_sd_sum_1",
        "weighted_sd_sum_2",
        "variance_contribution",
    ]
    return pd.concat(tables, ignore_index=True)[columns]


def _pair_table(coefficients: pd.DataFrame, weighted_sd_sum: pd.Series) -> pd.DataFrame:
    """One row per pair of elements with its contribution to the variance."""
    weighted_sd_sum = weighted_sd_sum.loc[coefficients.index]
    contributions = coefficients * np.outer(weighted_sd_sum, weighted_sd_sum)

    table = coefficients.stack().rename("correlation").to_frame()
    table.index.names = ["element_1", "element_2"]
    table["weighted_sd_sum_1"] = weighted_sd_sum.loc[
        table.index.get_level_values("element_1")
    ].values
    table["weighted_sd_sum_2"] = weighted_sd_sum.loc[
        table.index.get_level_values("element_2")
    ].values
    table["variance_contribution"] = contributions.stack().values
    return table.reset_index()


def _common_correlation(
    source: VarianceSource, include_correlation: bool
) -> pd.DataFrame:
    """Return ``C[e, e'] = R[e, e'] · √(ρ_s[e] ρ_s[e'])``."""
    correlation = _correlation_used(source, include_correlation)
    weight = np.sqrt(source.spatial_correlation.loc[correlation.index].values)
    return correlation * np.outer(weight, weight)


def _local_correlation(
    source: VarianceSource, include_correlation: bool
) -> pd.DataFrame:
    """Return ``L[e, e'] = R[e, e'] · √((1 - ρ_s[e])(1 - ρ_s[e']))`` for local elements."""
    elements = source.local_elements
    correlation = _correlation_used(source, include_correlation).loc[elements, elements]
    weight = np.sqrt(1 - source.spatial_correlation.loc[elements].values)
    return correlation * np.outer(weight, weight)


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
