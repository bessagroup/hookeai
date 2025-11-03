Installation
============

HookeAI can be easily installed using the `conda` package manager, which ensures that all dependencies are properly isolated.  
If you do not have `conda` installed, please follow the `official installation guide <https://www.anaconda.com/docs/getting-started/miniconda/install>`_.

Once `conda` is available, follow the steps below.

----

1. Clone the repository
-----------------------

Clone the HookeAI repository to your local directory:

.. code-block:: bash

   git clone git@github.com:bessagroup/hookeai.git

----

2. Create the Conda environment
-------------------------------

Navigate to the repository directory and create the HookeAI environment. This environment is setup in ``environment.yml`` and includes all required dependencies **except PyTorch**, which must be installed manually.

.. code-block:: bash

   cd hookeai
   conda env create -f environment.yml
   conda activate hookeai_env

----

2b. (Optional) Pip-only installation
------------------------------------

If you prefer not to use Conda, you can install all Python dependencies (**except PyTorch**) from ``requirements.txt``:

.. code-block:: bash

   pip install -r requirements.txt

----

3. Install PyTorch
------------------

HookeAI supports both CPU and CUDA installations. Please install PyTorch and its related libraries according to your system configuration by following the `official instructions <https://pytorch.org/get-started/locally/>`_.


----

4. Install HookeAI
------------------

Install the HookeAI package according to your needs:

- **Basic user:**

  .. code-block:: bash

     pip install .

- **Developer (editable mode):**

  .. code-block:: bash

     pip install -e .

----

5. Verify the installation
--------------------------

Check that HookeAI and PyTorch were successfully installed:

.. code-block:: bash

   python -c "import torch, hookeai; print('HookeAI and PyTorch installed successfully!')"





