Data generation
===============

HookeAI includes tools to **generate synthetic data sets** for material modeling research. 

----

Key resources
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - **Source file/directory**
     - **Description**
   * - ``hookeai/data_generation/strain_paths``
     - Directory containing the implementation of different **strain loading path generators**.
   * - ``hookeai/data_generation/spdg``
     - Directory containing the **Stochastic Patch Deformation Generator (SPDG)** to generate material patches subject to random boundary deformations.
   * - ``hookeai/user_scripts/synthetic_data/gen_response_dataset.py``
     - **Pre-configured user script** to generate synthetic strain-stress data sets from given material model.
   * - ``hookeai/user_scripts/synthetic_data/gen_random_specimen.py``
     - **Pre-configured user script** to generate data set of material patches undergoing random boundary deformations.

|

----

Strain loading paths
--------------------

A **diverse data set of strain loading paths** is fundamental to ensure an effective material model updating process. In the case of conventional (physics-based) material models, it is essential to accurately identify the model parameters. In the case of neural network (data-driven) and hybrid material models, a rich data set is crucial to avoid overfitting and to ensure good generalization performance beyond the training data.

HookeAI provides an interface (:code:`StrainPathGenerator`) to **generate different types of strain loading paths**. Two particular types are provided out-of-the-box: (i) Random (**polynomial**) strain loading paths (:code:`RandomStrainPathGenerator`), and (ii) Random (**proportional**) strain loading paths (:code:`ProportionalStrainPathGenerator`). Besides the different configuration options available for each type of generator, HookeAI also provides means to inject different types of **artifical noise** (:code:`NoiseGenerator`).

The strain loading paths can then be coupled with an available material model to compute the corresponding stress response, thus creating a synthetic strain-stress data set (:code:`MaterialResponseDatasetGenerator`) suitable for local model updating. Alternatively, they can serve as input to perform numerical simulations over representative volume elements of the material microstructure, which requires an external multi-scale solver.

.. image:: ../../../media/schematics/hookeai_strain_paths.png
   :width: 90 %
   :align: center


----

Material patches
----------------

HookeAI includes a method called `Stochastic Patch Deformation Generator (SPDG) <https://doi.org/10.1016/j.jmps.2025.106408>`_ to generate **synthetic data sets of material patches** discretized in a finite element mesh and subject to **random boundary deformations** (:code:`FiniteElementPatch`, :code:`FiniteElementPatchGenerator`).

These material patches can be directly translated into a finite element simulation with Dirichlet boundary conditions, which can be solved with an external solver to **obtain structural level data** (e.g., displacements, internal forces, reaction forces) as well as **material level data** (e.g., strains, stresses, internal variables from integration points). Hence, SPDG can be leveraged to generate a wide variety of synthetic data sets to support the development of **data-driven material and/or structural models**.

.. image:: ../../../media/schematics/hookeai_material_patches.png
   :width: 90 %
   :align: center

|

.. note ::

    HookeAI **does not** include a built-in finite element solver to perform the simulations over the generated material patches with SPDG. The material patch object (:code:`FiniteElementPatch`) contains all the required information to set up the corresponding **finite element simulation with Dirichlet boundary conditions**, which must be performed with an **external solver**. 
    

----

Pre-configured user scripts
---------------------------
HookeAI provides a set of **pre-configured user scripts** to generate different types of synthetic data sets. These scripts can be readily used and demonstrate the typical workflow for **generating synthetic data**, including all pre- and post-processing steps such as configuring the generator parameters, setting up the material model, and exporting the generated data sets. These can also be **easily adapted** to suit specific user needs, without the need to implement the entire workflow from scratch.

