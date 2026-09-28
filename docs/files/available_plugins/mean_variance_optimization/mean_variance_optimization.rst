:orphan:

.. _available_plugins.mean_variance_optimization:

Mean variance optimization plugin
----------------------------------------------

The ``mean_variance_optimization`` plugin replaces the objective of ZEN-garden (the
net present cost) by a mean-variance objective. The variance stems from uncertain
technology capex and import prices. It subscribes to two events:

- ``after_model_construction``: replaces the objective by
  ``net_present_cost + weighting_factor * variance`` (method ``weighting_factor``) or
  by ``net_present_cost + regularization_factor * sum(x^2)`` (method
  ``regularization_only``).
- ``after_postprocessing``: computes the variance of the solution and writes it,
  together with the objective value and the net present cost, to
  ``mean_variance_dict.json`` in the output folder.

The relative standard deviations and correlations are read from the dataset:

.. code-block:: text

    <dataset>/mean_variance/technology_capex/sd.csv
    <dataset>/mean_variance/technology_capex/correlation.csv
    <dataset>/mean_variance/imports/sd.csv
    <dataset>/mean_variance/imports/correlation.csv

The plugin is activated in the config file:

.. code-block:: yaml

    plugins:
      mean_variance_optimization:
        method: weighting_factor
        weighting_factor: 1.0e-8
        include_correlation: true
        include_variances_for: ["technology_capex", "imports"]

Module documentation
^^^^^^^^^^^^^^^^^^^^

.. automodule:: zen_garden_plugins.mean_variance_optimization.plugin
   :members:
   :undoc-members:
