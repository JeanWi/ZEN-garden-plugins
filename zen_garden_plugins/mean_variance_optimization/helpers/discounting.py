"""Weights that turn a cost of one model entry into its share of the NPC.

The variance term must describe the same quantity as the mean term, the net
present cost (NPC). The standard deviations are therefore multiplied with the
same annuity and discount factors that ZEN-garden applies to the costs. The
functions below mirror these ZEN-garden constraints:

- ``NetPresentCostConstraint``
  (``zen_garden/elements/energy_system/constraints/net_present_cost_constraint.py``)
- ``CostCapexYearlyConstraint`` and ``TechnologyConstraint.get_lifetime_range``
  (``zen_garden/elements/technology/constraints/``)
- ``CostCarrierTotalConstraint``
  (``zen_garden/elements/carrier/constraints/cost_carrier_total_constraint.py``)

If these formulas change in ZEN-garden, the functions here must be updated as
well. The tests ``test_capex_npv_weight_reproduces_net_present_cost`` and
``test_import_npv_weight_reproduces_net_present_cost`` check this.
"""

import numpy as np
import xarray as xr


def discount_factor_per_year(optimization_model) -> xr.DataArray:
    """Return the factor with which the total cost of a modeled year enters the NPC.

    A modeled year ``y`` represents ``Δy = interval_between_years`` calendar
    years (only one for the last year of the horizon), which are all discounted::

        factor[y] = Σ_{i=0}^{Δy-1} (1 / (1 + r)) ** (Δy · (y - y_0) + i)

    Returns:
        Discount factor with dimension ``set_years``.
    """
    model_schema = optimization_model.model_schema
    discount_rate = float(optimization_model.parameters.discount_rate)
    interval_between_years = optimization_model.config.system.interval_between_years
    first_year = model_schema.set_years[0]
    last_year_of_horizon = model_schema.set_years_entire_horizon[-1]

    factors = []
    for year in model_schema.set_years:
        if year == last_year_of_horizon:
            calendar_years_represented = 1
        else:
            calendar_years_represented = interval_between_years
        factors.append(
            sum(
                (1 / (1 + discount_rate))
                ** (interval_between_years * (year - first_year) + calendar_year)
                for calendar_year in range(calendar_years_represented)
            )
        )
    return xr.DataArray(
        factors, coords={"set_years": list(model_schema.set_years)}, dims="set_years"
    )


def annuity_factor(discount_rate: float, depreciation_time: float) -> float:
    """Return the share of the overnight capex that is paid in every year.

    ``a = (1 + r)^L · r / ((1 + r)^L - 1)``, or ``1 / L`` if ``r = 0``.
    """
    if discount_rate == 0:
        return 1 / depreciation_time
    growth = (1 + discount_rate) ** depreciation_time
    return growth * discount_rate / (growth - 1)


def capex_npv_weight(optimization_model) -> xr.DataArray:
    """Return the NPC of one unit of overnight capex, per technology and build year.

    Capex spent in the build year ``y`` is paid as an annuity ``a`` in every
    modeled year ``y'`` of the depreciation period that lies inside the horizon,
    i.e. ``y <= y' <= y + floor(L / Δy) - 1``. Each payment is discounted::

        weight[tech, y] = a_tech · Σ_{y'} factor[y']

    Returns:
        Weight with dimensions ``set_technologies`` and ``set_years``.
    """
    discount_rate = float(optimization_model.parameters.discount_rate)
    interval_between_years = optimization_model.config.system.interval_between_years
    depreciation_time = optimization_model.parameters.depreciation_time
    discount_factor = discount_factor_per_year(optimization_model)
    years = list(discount_factor["set_years"].values)

    technologies = list(depreciation_time["set_technologies"].values)
    weight = xr.DataArray(
        0.0,
        coords={"set_technologies": technologies, "set_years": years},
        dims=["set_technologies", "set_years"],
    )
    for technology in technologies:
        lifetime = float(depreciation_time.sel(set_technologies=technology))
        annuity = annuity_factor(discount_rate, lifetime)
        # number of further modeled years in which the annuity is paid
        # (same conservative rounding as in ZEN-garden)
        further_years_paid = int(np.floor(lifetime / interval_between_years)) - 1
        for build_year in years:
            payment_years = [
                year
                for year in years
                if build_year <= year <= build_year + further_years_paid
            ]
            weight.loc[technology, build_year] = annuity * float(
                discount_factor.sel(set_years=payment_years).sum()
            )
    return weight


def operation_npv_weight(optimization_model, time_steps) -> xr.DataArray:
    """Return the NPC of a cost of one unit per hour in an operational time step.

    An operational time step ``t`` of year ``y`` represents ``duration[t]`` hours
    and its cost is discounted with the factor of its year::

        weight[t] = duration[t] · factor[y(t)]

    Args:
        optimization_model: ZEN-garden optimization model.
        time_steps: ZEN-garden time steps, used to find the year of each
            operational time step.

    Returns:
        Weight with dimension ``set_time_steps_operation``.
    """
    duration = optimization_model.parameters.time_steps_operation_duration
    discount_factor = discount_factor_per_year(optimization_model)

    weight = duration * 0.0
    for year in discount_factor["set_years"].values:
        time_steps_of_year = list(time_steps.get_time_steps_year2operation(year))
        weight.loc[time_steps_of_year] += (
            float(discount_factor.sel(set_years=year))
            * duration.loc[time_steps_of_year]
        )
    return weight
