Data generation
===============

HookeAI includes tools to generate synthetic data sets for material modeling research. 

----

Strain loading paths
--------------------

A diverse data set of strain loading paths is fundamental to ensure an effective material model updating process. In the case of conventional (physics-based) material models, it is essential to accurately identify the model parameters. In the case of neural network (data-driven) and hybrid material models, a rich data set is crucial to avoid overfitting and to ensure good generalization performance beyond the training data.

HookeAI provides an interface (:code:`StrainPathGenerator`) to generate different types of strain loading paths. Two particular types are provided out-of-the-box: (i) Random (polynomial) strain loading paths (:code:`RandomStrainPathGenerator`), and (ii) Random (proportional) strain loading paths (:code:`ProportionalStrainPathGenerator`). Besides the different configuration options available for each type of generator, HookeAI also provides means to inject different types of artifical noise (:code:`NoiseGenerator`).

The strain loading paths can then be coupled with an available material model to compute the corresponding stress response, thus creating a synthetic strain-stress data set (:code:`MaterialResponseDatasetGenerator`) suitable for local model updating. Alternatively, they can serve as input to perform numerical simulations over representative volume elements of the material microstructure, which requires an external multi-scale solver.

----

Material patches
----------------

HookeAI includes a method called `Stochastic Patch Deformation Generator (SPDG) <https://arxiv.org/abs/2505.07801>`_ to generate synthetic data sets of material patches discretized in a finite element mesh and subject to random boundary deformations (:code:`FiniteElementPatch`, :code:`FiniteElementPatchGenerator`).

These material patches can be directly translated into a finite element simulation with Dirichlet boundary conditions, which can be solved with an external solver to obtain structural level data (e.g., displacements, internal forces, reaction forces) as well as material level data (e.g., strains, stresses, internal variables from integration points). Hence, SPDG can be leveraged to generate a wide variety of synthetic data sets to support the development of data-driven material and/or structural models.
