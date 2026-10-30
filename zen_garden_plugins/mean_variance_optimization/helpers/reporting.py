"""Write the mean and variance of the solution to the output folder.

Two files are written next to the standard ZEN-garden results:

- ``mean_variance.json``: mean (NPC), variance and standard deviation of the
  NPC, the objective value and the settings used.
- ``mean_variance_contributions.csv``: contribution of every pair of elements
  to the variance (columns described in
  :func:`~zen_garden_plugins.mean_variance_optimization.variance_term.variance_from_solution`).
"""

import json

import pandas as pd

from zen_garden_plugins.mean_variance_optimization.helpers.variance_sources import (
    create_variance_sources,
)
from zen_garden_plugins.mean_variance_optimization.helpers.variance_term import (
    variance_from_solution,
)


def write_mean_variance_report(postprocessing, settings: dict) -> None:
    """Compute the variance of the solved model and write it to files."""
    optimization_model = postprocessing.optimization_model
    lp_model = optimization_model.lp_model

    sources = create_variance_sources(
        optimization_model, postprocessing.time_steps, settings
    )
    contributions = pd.concat(
        [
            variance_from_solution(
                optimization_model, source, settings["include_correlation"]
            ).assign(source=source.name)
            for source in sources
        ]
        or [pd.DataFrame(columns=["source", "variance_contribution"])],
        ignore_index=True,
    )
    variance_per_source = (
        contributions.groupby("source")["variance_contribution"].sum().to_dict()
    )
    variance = float(sum(variance_per_source.values()))

    report = {
        "mean_npc": float(
            lp_model.variables["net_present_cost"].solution.sum("set_years")
        ),
        "variance": variance,
        "standard_deviation": variance**0.5,
        "variance_per_source": variance_per_source,
        "objective_value": float(lp_model.objective.value),
        "settings": settings,
    }

    with open(postprocessing.name_dir / "mean_variance.json", "w") as file:
        json.dump(report, file, indent=2)
    columns = ["source"] + [c for c in contributions.columns if c != "source"]
    contributions[columns].to_csv(
        postprocessing.name_dir / "mean_variance_contributions.csv", index=False
    )
