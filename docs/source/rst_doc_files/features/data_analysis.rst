Data analysis and visualization tools
=====================================

**Data analysis and visualization** tools are essential to understand material response data sets and evaluate the performance of material models. For this reason, HookeAI provides a complete toolkit to streamline the generation of **insightful visualizations** and provide **material-related data analysis**.

----

Key resources
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - **Source file/directory**
     - **Description**
   * - ``hookeai/ioput/plots.py``
     - Wide variety of plotting functions suitable for **general visualization of numerical data**.
   * - ``hookeai/model_architectures/model_prediction.py``
     - Essential prediction plots for **generic time series data**.
   * - ``hookeai/model_architectures/convergence_plots.py``
     - **Convergence plots** with respect to the training data set size, namely the prediction loss and generic time series predictions.
   * - ``hookeai/utilities/output_prediction_metrics.py``
     - Computation and plots of **mean predictions metrics** for generic time series data.
   * - ``hookeai/miscellaneous/materials/compare_material_models.py``
     - **Stress prediction comparison** between different material models on given strain-stress data set samples.
   * - ``hookeai/miscellaneous/materials/compare_hardening_laws.py``
     - **Comparison of strain hardening laws** for elastoplastic material models.
   * - ``hookeai/miscellaneous/materials/plot_yield_surface.py``
     - **Visualization of yield surfaces** of elastoplastic material models.
   * - ``hookeai/user_scripts/synthetic_data/gen_response_dataset.py``
     - Complete set of plots to describe **strain-stress data sets**.



