"""Mean-variance optimization plugin for ZEN-garden.

Replaces the objective of ZEN-garden (net present cost) by a mean-variance
objective, where the variance stems from uncertain technology capex and import
prices. The relative standard deviations and correlations are read from the
dataset folder ``mean_variance/<variance_type>/{sd,correlation}.csv``.

After postprocessing, the variance of the solution is written to
``mean_variance_dict.json`` in the output folder.
"""

from typing import TYPE_CHECKING, Literal

import pandas as pd
from tqdm import tqdm
from zen_garden import (  # type: ignore[import-untyped]
    ConfigBase,
    Event,
    EventPublisher,
)

from zen_garden_plugins.mean_variance_optimization.helpers import (
    CovarianceImports,
    CovarianceTechnologies,
    get_non_zero_elements,
)

if TYPE_CHECKING:
    from zen_garden.model.optimization_model import (  # type: ignore[import-untyped]
        OptimizationModel,
    )
    from zen_garden.postprocess.postprocess import (  # type: ignore[import-untyped]
        Postprocess,
    )

PLUGIN_NAME = "mean_variance_optimization"


class Config(ConfigBase):
    """Configuration for the mean-variance optimization plugin.

    Users can override these defaults in their config file:

        plugins:
          mean_variance_optimization:
            weighting_factor: 1.0e-8
            include_variances_for: ["technology_capex", "imports"]
    """

    #: ``weighting_factor``: minimize NPC + weighting_factor * variance;
    #: ``regularization_only``: minimize NPC + regularization_factor * sum(x^2)
    method: Literal["weighting_factor", "regularization_only"] = "weighting_factor"
    weighting_factor: float | None = None
    include_correlation: bool = True
    regularization_factor: float = 1e-8
    include_variances_for: list[Literal["technology_capex", "imports"]] = [
        "technology_capex",
        "imports",
    ]


def _get_settings(config) -> dict | None:
    """Return the validated plugin settings from the ZEN-garden config.

    Returns None if the plugin is not configured for this run (the event handlers
    stay registered once the module has been imported).
    """
    return config.plugins.get(PLUGIN_NAME)


def _add_aggregate_variable(optimization_model, name, agg_index, doc):
    """Add an auxiliary variable indexed by ``agg_index`` to the model.

    The variable is registered with a doc string and without unit in the variable
    registry of ZEN-garden so that it is saved in the postprocessing.
    """
    optimization_model.lp_model.add_variables(name=name, coords=[agg_index])
    optimization_model.variables.docs[name] = doc
    optimization_model.variables.units[name] = None


def _technology_correlation(optimization_model, quadratic_term, settings):
    """Simplified variance term: aggregate capacity additions over locations and time steps,
    and compute correlations only per technology pair (not per location/time).

    Introduces an auxiliary variable ``capacity_addition_tech_agg`` (one per technology /
    capacity-type pair) that equals the sum of ``capacity_addition`` over all locations and
    yearly time steps, and constrains it accordingly.  The quadratic variance term is then
    built from products of these scalar variables, which linopy can handle as a proper QP.
    """
    model = optimization_model.lp_model

    # Get covariance matrix
    covariance_calculation = CovarianceTechnologies(optimization_model)
    covariance_matrix_indexmap, covariance_matrix  = covariance_calculation.generate_covariance_matrix()

    # Get non-zero entries
    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)

    # Construct variables (only have integer indexing)
    agg_index = pd.Index(
        range(len(covariance_matrix_indexmap)),
        name="agg_index"
    )
    inverse_index_map = {
        v: k
        for k, v in covariance_matrix_indexmap.items()
    }

    _add_aggregate_variable(
        optimization_model,
        "capacity_addition_tech_agg",
        agg_index,
        "Capacity addition aggregated per entry of the capex covariance matrix",
    )

    include_correlation = settings["include_correlation"]

    # Construct constraints for aggregate variables
    for variable_index, variable_key in tqdm(inverse_index_map.items(), total=len(inverse_index_map),
                       desc="Constructing aggregate variables (technology correlation"):


        technology = variable_key[0]
        location = variable_key[1]
        year = variable_key[2]
        capacity_type = variable_key[3]

        selection_dict, sum_list = covariance_calculation.generate_sum_list(technology, location, year, capacity_type)

        lhs_exp = model.variables["capacity_addition_tech_agg"].sel(agg_index=variable_index)
        rhs_exp = model.variables["capacity_addition"].sel(selection_dict).sum(sum_list)
        constraint_capacity_addition = lhs_exp == rhs_exp

        optimization_model.add_constraint(
            f"constraint_capacity_addition_tech_agg{variable_index}", constraint_capacity_addition
        )

    # Construct quadratic term
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Constructing quadratic variance term (technology correlation)"):
        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]

        C_i = model.variables["capacity_addition_tech_agg"].sel(agg_index=index_i)
        C_j = model.variables["capacity_addition_tech_agg"].sel(agg_index=index_j)

        covariance = covariance_matrix[index_i, index_j]

        if index_i != index_j:
            factor = 2
            if include_correlation:
                quadratic_term += factor * covariance * C_i * C_j
        else:
            factor = 1
            quadratic_term += factor * covariance * C_i * C_j

    return quadratic_term

def _import_correlation(optimization_model, quadratic_term, settings):

    model = optimization_model.lp_model

    covariance_calculation = CovarianceImports(optimization_model)
    covariance_matrix_indexmap, covariance_matrix = covariance_calculation.generate_covariance_matrix()

    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)

    agg_index = pd.Index(
            range(len(covariance_matrix_indexmap)),
            name="agg_index"
        )
    inverse_index_map = {
        v: k
        for k, v in covariance_matrix_indexmap.items()
    }

    _add_aggregate_variable(
        optimization_model,
        "imports_agg",
        agg_index,
        "Carrier import aggregated per entry of the import price covariance matrix",
    )

    include_correlation = settings["include_correlation"]

    # Construct constraints for aggregate variables
    for variable_index, variable_key in tqdm(inverse_index_map.items(), total=len(inverse_index_map),
                       desc="Constructing aggregate variables (import cost correlation)"):

        carrier = variable_key[0]
        location = variable_key[1]
        time_step_operation = variable_key[2]

        selection_dict, sum_list = covariance_calculation.generate_sum_list(carrier, location, time_step_operation)

        lhs_exp = model.variables["imports_agg"].sel(agg_index=variable_index)
        rhs_exp = model.variables["flow_import"].sel(selection_dict).sum(sum_list)
        constraint_capacity_addition = lhs_exp == rhs_exp

        optimization_model.add_constraint(
            f"constraint_imports_agg{variable_index}", constraint_capacity_addition
        )

    # Construct quadratic term
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Constructing quadratic variance term (import cost correlation)"):
        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]

        C_i = model.variables["imports_agg"].sel(agg_index=index_i)
        C_j = model.variables["imports_agg"].sel(agg_index=index_j)

        covariance = covariance_matrix[index_i, index_j]

        if index_i != index_j:
            factor = 2
            if include_correlation:
                quadratic_term += factor * covariance * C_i * C_j
        else:
            factor = 1
            quadratic_term += factor * covariance * C_i * C_j

    return quadratic_term



def _create_diagonal_squared_terms(optimization_model, settings):
    model = optimization_model.lp_model
    regularization_term = 0
    regularization_factor = settings["regularization_factor"]
    for var in model.variables:
        regularization_term += regularization_factor * (model.variables[var] * model.variables[var]).sum()
    return regularization_term

def _postprocess_technology_variance(postprocessing, variance_cumsum, settings):
    optimization_model = postprocessing.optimization_model
    covariance_calculation = CovarianceTechnologies(optimization_model)
    covariance_matrix_indexmap, covariance_matrix  = covariance_calculation.generate_covariance_matrix()
    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)
    covariance_rows = []

    include_correlation = settings["include_correlation"]

    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Postprocessing variance (technology correlation)"):

        technology_i = pair[0][0]
        technology_j = pair[1][0]

        location_i = pair[0][1]
        location_j = pair[1][1]

        year_i = pair[0][2]
        year_j = pair[1][2]

        capacity_type_i = pair[0][3]
        capacity_type_j = pair[1][3]

        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]
        covariance = covariance_matrix[index_i, index_j]

        selection_dict_i, sum_list_i = covariance_calculation.generate_sum_list(technology_i, location_i, year_i, capacity_type_i)
        selection_dict_j, sum_list_j = covariance_calculation.generate_sum_list(technology_j, location_j, year_j, capacity_type_j)

        C_i = float(optimization_model.lp_model.variables["capacity_addition"].solution.sel(selection_dict_i).sum(sum_list_i))
        C_j = float(optimization_model.lp_model.variables["capacity_addition"].solution.sel(selection_dict_j).sum(sum_list_j))


        if index_i != index_j:
            factor = 2
            if include_correlation:
                variance_cumsum += factor * covariance * C_i * C_j
        else:
            factor = 1
            variance_cumsum += factor * covariance * C_i * C_j


        covariance_rows.append({
            "pair": pair,
            "covariance": covariance * C_i * C_j,
            "covariance_factor": covariance
        })

    pd.DataFrame(covariance_rows).to_csv(
        postprocessing.name_dir / "covariance_pairs_reporting.csv", index=False
    )

    return variance_cumsum

def _postprocess_imports_variance(postprocessing, variance_cumsum, settings):
    optimization_model = postprocessing.optimization_model

    include_correlation = settings["include_correlation"]

    covariance_calculation = CovarianceImports(optimization_model)
    covariance_matrix_indexmap, covariance_matrix = covariance_calculation.generate_covariance_matrix()

    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)

    # Construct quadratic term
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Constructing quadratic variance term (import cost correlation)"):
        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]

        carrier_i = pair[0][0]
        carrier_j = pair[1][0]

        location_i = pair[0][1]
        location_j = pair[1][1]

        time_step_operation_i = pair[0][2]
        time_step_operation_j = pair[1][2]

        covariance = covariance_matrix[index_i, index_j]


        selection_dict_i, sum_list_i = covariance_calculation.generate_sum_list(carrier_i, location_i, time_step_operation_i)
        selection_dict_j, sum_list_j = covariance_calculation.generate_sum_list(carrier_j, location_j, time_step_operation_j)
        C_i = float(optimization_model.lp_model.variables["flow_import"].solution.sel(selection_dict_i).sum(sum_list_i))
        C_j = float(optimization_model.lp_model.variables["flow_import"].solution.sel(selection_dict_j).sum(sum_list_j))

        if index_i != index_j:
            factor = 2
            if include_correlation:
                variance_cumsum += factor * covariance * C_i * C_j
        else:
            factor = 1
            variance_cumsum += factor * covariance * C_i * C_j


    return variance_cumsum


@EventPublisher.register(Event.after_postprocessing)
def calculate_variance_from_solution(postprocessing: "Postprocess") -> None:
    """Compute the realized capex variance from the solved capacity_addition values.

    Variance = Σ_{i,j} ρ_{ij} · σ_i · σ_j · C_i · C_j

    where C_k = Σ_{loc, t} capacity_addition[tech_k, cap_k, loc, t]  (from solution).

    The scalars are written to ``mean_variance_dict.json`` in the output folder of
    the postprocessing.
    """
    settings = _get_settings(postprocessing.config)
    if settings is None:
        return

    # Get covariance matrix
    variance_cumsum = 0.0

    if "technology_capex" in settings["include_variances_for"]:
        variance_cumsum = _postprocess_technology_variance(postprocessing, variance_cumsum, settings)
    if "imports" in settings["include_variances_for"]:
        variance_cumsum = _postprocess_imports_variance(postprocessing, variance_cumsum, settings)

    plugin_reporting = {}
    plugin_reporting["variance"]= variance_cumsum
    plugin_reporting["standard_deviation"]= variance_cumsum ** 0.5


    model = postprocessing.optimization_model.lp_model
    objective_value = float(model.objective.value)
    npv_value = float(model.variables["net_present_cost"].solution.sum("set_years"))

    plugin_reporting["objective_value"] = objective_value
    plugin_reporting["npv"] = npv_value
    plugin_reporting["weighting_factor"] = settings["weighting_factor"]

    postprocessing._write_json_file(postprocessing.name_dir.joinpath("mean_variance_dict.json"), plugin_reporting)


@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_model: "OptimizationModel") -> None:
    """Replace the net-present-cost objective by the mean-variance objective."""
    settings = _get_settings(optimization_model.config)
    if settings is None:
        return

    variance_term = 0

    model = optimization_model.lp_model

    quadratic_objective = settings["method"] == "regularization_only" or bool(
        settings["weighting_factor"]
    )
    if quadratic_objective and optimization_model.config.solver.use_scaling:
        raise ValueError(
            "The scaling of ZEN-garden does not support the quadratic objective of "
            "the mean-variance optimization. Set solver.use_scaling to false."
        )

    model.remove_objective()
    npv_term = model.variables["net_present_cost"].sum("set_years")
    method = settings["method"]

    if method == "weighting_factor":
        weighting_factor = settings["weighting_factor"]
        if weighting_factor is None: weighting_factor = 0

        if weighting_factor != 0:
            if "technology_capex" in settings["include_variances_for"]:
                variance_term = _technology_correlation(optimization_model, variance_term, settings)
            if "imports" in settings["include_variances_for"]:
                variance_term = _import_correlation(optimization_model, variance_term, settings)
            objective = weighting_factor * variance_term + npv_term
        else:
            objective = npv_term

        sense = "min"
        model.add_objective(objective, sense=sense)

    elif method == "regularization_only":
        regularization_term = _create_diagonal_squared_terms(optimization_model, settings)
        objective = regularization_term + npv_term
        sense = "min"
        model.add_objective(objective, sense=sense)
