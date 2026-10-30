"""Uncertain cost components ("variance sources") of the mean-variance plugin.

Each source describes one uncertain cost that enters the net present cost (NPC)
linearly, ``cost = Σ_i c_i · x_i``, where ``x`` is a decision variable and ``c``
an uncertain cost parameter:

- ``technology_capex``: ``x = capacity_addition``, ``c = specific capex``
- ``imports``: ``x = flow_import``, ``c = price_import``

For every entry ``i`` of ``x`` the absolute standard deviation of its NPC
coefficient is::

    σ_i = relative_sd[element(i)] · c_i · npv_weight_i

where ``npv_weight`` applies the same annuity and discounting as ZEN-garden
(see ``discounting.py``).

Correlation assumption: the correlation matrix is given per element
(technology or carrier). All entries of the same element (all locations,
years, capacity types or time steps) are assumed to be perfectly correlated
(ρ = 1). They can therefore be combined exactly into one weighted sum per
element, ``y_e = Σ_{i of e} σ_i · x_i``, and the variance of the NPC is::

    Var = Σ_e Σ_e' R[e, e'] · y_e · y_e'
"""

from dataclasses import dataclass

import pandas as pd
import xarray as xr

from zen_garden_plugins.mean_variance_optimization.helpers.discounting import (
    capex_npv_weight,
    operation_npv_weight,
)
from zen_garden_plugins.mean_variance_optimization.helpers.read_data import (
    get_input_folder,
    read_correlation_matrix,
    read_relative_sd,
)

TECHNOLOGY_CAPEX = "technology_capex"
IMPORTS = "imports"


@dataclass
class VarianceSource:
    """All data needed to build the variance term of one uncertain cost.

    Attributes:
        name: Name of the source in the plugin config and the input folder.
        decision_variable: Name of the ZEN-garden variable ``x`` whose cost is
            uncertain.
        element_dimension: Dimension of ``x`` that holds the elements of the
            correlation matrix (technologies or carriers).
        absolute_sd: Absolute standard deviation ``σ`` of the NPC coefficient
            of every entry of ``x`` (same dimensions as ``x``).
        correlation: Correlation matrix of the uncertain elements.
    """

    name: str
    decision_variable: str
    element_dimension: str
    absolute_sd: xr.DataArray
    correlation: pd.DataFrame

    @property
    def weighted_sum_variable(self) -> str:
        """Name of the auxiliary variable ``y_e = Σ_{i of e} σ_i · x_i``."""
        return f"{self.decision_variable}_weighted_sd_sum"

    @property
    def uncertain_elements(self) -> list[str]:
        """Elements with a nonzero standard deviation."""
        return list(self.correlation.index)

    @property
    def summed_dimensions(self) -> list[str]:
        """Dimensions of ``x`` that are summed up in the weighted sum."""
        return [dim for dim in self.absolute_sd.dims if dim != self.element_dimension]


def create_variance_sources(
    optimization_model, time_steps, settings: dict
) -> list[VarianceSource]:
    """Create the variance sources selected in ``include_variances_for``."""
    builders = {
        TECHNOLOGY_CAPEX: lambda: create_technology_capex_source(
            optimization_model, settings
        ),
        IMPORTS: lambda: create_import_price_source(
            optimization_model, time_steps, settings
        ),
    }
    return [builders[name]() for name in settings["include_variances_for"]]


def create_technology_capex_source(
    optimization_model, settings: dict
) -> VarianceSource:
    """Variance source for the specific capex of all technologies.

    Not included: the distance-dependent capex of transport technologies
    (``capex_per_distance_transport``), because it is linked to a binary
    build decision and not to ``capacity_addition``.
    """
    capacity_addition = optimization_model.variables["capacity_addition"]
    technologies = list(optimization_model.sets["set_technologies"])
    folder = get_input_folder(
        optimization_model.config.analysis.dataset, TECHNOLOGY_CAPEX
    )

    relative_sd = read_relative_sd(folder, "technology", technologies)
    relative_sd = xr.DataArray(relative_sd.rename_axis("set_technologies"))
    absolute_sd = (
        relative_sd
        * _specific_capex(optimization_model)
        * capex_npv_weight(optimization_model)
    )
    absolute_sd = _align_to_variable(absolute_sd, capacity_addition)
    return _create_source(
        TECHNOLOGY_CAPEX,
        "capacity_addition",
        "set_technologies",
        absolute_sd,
        folder,
        settings,
    )


def create_import_price_source(
    optimization_model, time_steps, settings: dict
) -> VarianceSource:
    """Variance source for the import prices of all carriers.

    The NPC of an import is ``price_import · flow_import · duration ·
    discount factor``; export revenues are not uncertain in this plugin.
    """
    flow_import = optimization_model.variables["flow_import"]
    carriers = list(optimization_model.sets["set_carriers"])
    folder = get_input_folder(optimization_model.config.analysis.dataset, IMPORTS)

    relative_sd = read_relative_sd(folder, "carrier", carriers)
    relative_sd = xr.DataArray(relative_sd.rename_axis("set_carriers"))
    absolute_sd = (
        relative_sd
        * optimization_model.parameters.price_import
        * operation_npv_weight(optimization_model, time_steps)
    )
    absolute_sd = _align_to_variable(absolute_sd, flow_import)
    return _create_source(
        IMPORTS, "flow_import", "set_carriers", absolute_sd, folder, settings
    )


def _create_source(
    name, decision_variable, element_dimension, absolute_sd, folder, settings
) -> VarianceSource:
    """Keep only uncertain elements and read their correlation matrix."""
    summed_dimensions = [dim for dim in absolute_sd.dims if dim != element_dimension]
    is_uncertain = (absolute_sd != 0).any(summed_dimensions)
    uncertain_elements = list(
        absolute_sd[element_dimension].values[is_uncertain.values]
    )

    correlation = read_correlation_matrix(
        folder, uncertain_elements, settings["correlation_regularization"]
    )
    return VarianceSource(
        name=name,
        decision_variable=decision_variable,
        element_dimension=element_dimension,
        absolute_sd=absolute_sd.sel({element_dimension: uncertain_elements}),
        correlation=correlation,
    )


def _specific_capex(optimization_model) -> xr.DataArray:
    """Combine the specific capex of all technology types into one array.

    The result has the dimensions of ``capacity_addition``: conversion and
    transport capex refer to the ``power`` capacity, storage capex to both
    ``power`` and ``energy``. Nodes and edges both become ``set_location``.
    """
    parameters = optimization_model.parameters
    conversion = parameters.capex_specific_conversion.rename(
        {"set_conversion_technologies": "set_technologies", "set_nodes": "set_location"}
    ).expand_dims(set_capacity_types=["power"])
    storage = parameters.capex_specific_storage.rename(
        {"set_storage_technologies": "set_technologies", "set_nodes": "set_location"}
    )
    transport = parameters.capex_specific_transport.rename(
        {"set_transport_technologies": "set_technologies", "set_edges": "set_location"}
    ).expand_dims(set_capacity_types=["power"])
    return xr.concat(
        [conversion, storage, transport], dim="set_technologies", join="outer"
    ).fillna(0.0)


def _align_to_variable(data: xr.DataArray, variable) -> xr.DataArray:
    """Give ``data`` exactly the coordinates and dimension order of ``variable``.

    Entries that do not exist in ``data`` (e.g. technologies without capex)
    get a standard deviation of 0.
    """
    coordinates = {dim: variable.coords[dim].values for dim in variable.dims}
    return (
        data.reindex(coordinates, fill_value=0.0).fillna(0.0).transpose(*variable.dims)
    )
