:orphan:

.. _available_plugins.mean_variance_optimization:

Mean variance optimization plugin
----------------------------------------------

The ``mean_variance_optimization`` plugin adds the **variance of the net present
cost (NPC)** to the optimization. The variance stems from uncertain technology
capex and uncertain carrier import prices.

Methods
^^^^^^^

The objective is selected with ``method``:

- ``weighting_factor``: minimize :math:`\mathrm{NPC} + w \cdot \mathrm{Var}`
- ``cost_constraint``: minimize :math:`\mathrm{Var}` subject to
  :math:`\mathrm{NPC} \le` ``cost_budget``
- ``regularization_only``: minimize :math:`\mathrm{NPC} + \varepsilon \sum x^2`
  over all variables (no variance; makes the solution unique)

How the variance is calculated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each uncertain cost enters the NPC linearly, :math:`\sum_i c_i x_i`, with

- technology capex: :math:`x` = ``capacity_addition``, :math:`c` = specific capex
  (``capex_specific_conversion``, ``capex_specific_storage``,
  ``capex_specific_transport``)
- imports: :math:`x` = ``flow_import``, :math:`c` = ``price_import``

The absolute standard deviation of the NPC coefficient of entry :math:`i` is

.. math::

    \sigma_i = \mathrm{relative\_sd}_{e(i)} \cdot c_i \cdot \omega_i

where :math:`e(i)` is the technology or carrier of entry :math:`i` and
:math:`\omega_i` is the same annuity and discount factor that ZEN-garden uses in
the NPC (for imports also the duration of the time step).

The correlation matrix :math:`R` is given per technology (and per carrier). All
entries of the same technology (all locations, years and capacity types) are
assumed to be perfectly correlated. They are combined exactly into one
auxiliary variable per technology,

.. math::

    y_e = \sum_{i \in e} \sigma_i x_i, \qquad
    \mathrm{Var} = \sum_{e} \sum_{e'} R_{e e'} \, y_e \, y_{e'}

With ``include_correlation: false``, only the diagonal of :math:`R` is used.
The distance-dependent capex of transport technologies
(``capex_per_distance_transport``) is not included.

.. note::
    The quadratic objective cannot be combined with the scaling of ZEN-garden.
    Set ``solver.use_scaling: false``.

Input data
^^^^^^^^^^

The input files are written by ZEN-Europe into the dataset:

.. code-block:: text

    <dataset>/mean_variance/technology_capex/relative_sd.csv   columns: technology, relative_sd
    <dataset>/mean_variance/technology_capex/correlation.csv   square matrix of technologies
    <dataset>/mean_variance/imports/relative_sd.csv            columns: carrier, relative_sd
    <dataset>/mean_variance/imports/correlation.csv            square matrix of carriers

``relative_sd`` is the standard deviation as a share of the cost parameter
(0.1 = 10 %). Every technology and carrier of the model must appear in the
files. The correlation matrix is matched by name, must be symmetric with ones
on the diagonal, and positive semi-definite.

Configuration
^^^^^^^^^^^^^

.. code-block:: yaml

    solver:
      use_scaling: false
    plugins:
      mean_variance_optimization:
        method: weighting_factor          # or cost_constraint, regularization_only
        weighting_factor: 1.0e-8
        cost_budget: null                 # required for cost_constraint
        include_variances_for: ["technology_capex", "imports"]
        include_correlation: true
        correlation_regularization: 1.0e-6
        regularization_factor: null     # only for regularization_only

Results
^^^^^^^

The plugin writes two files next to the standard results:

- ``mean_variance.json``: mean NPC, variance, standard deviation, variance per
  source, objective value and the settings used
- ``mean_variance_contributions.csv``: contribution of every pair of
  technologies (or carriers) to the variance

The auxiliary variables ``capacity_addition_weighted_sd_sum`` and
``flow_import_weighted_sd_sum`` are saved with the other variables.

Required ZEN-garden events
^^^^^^^^^^^^^^^^^^^^^^^^^^

The plugin uses the events ``after_model_construction`` (replaces the
objective) and ``after_postprocessing`` (writes the results).

Module documentation
^^^^^^^^^^^^^^^^^^^^

.. automodule:: zen_garden_plugins.mean_variance_optimization.plugin
   :members:
   :undoc-members:

.. automodule:: zen_garden_plugins.mean_variance_optimization.variance_sources
   :members:

.. automodule:: zen_garden_plugins.mean_variance_optimization.variance_term
   :members:

.. automodule:: zen_garden_plugins.mean_variance_optimization.discounting
   :members:

.. automodule:: zen_garden_plugins.mean_variance_optimization.input_data
   :members:
