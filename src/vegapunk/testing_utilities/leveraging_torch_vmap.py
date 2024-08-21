# Standard
import os
import psutil
import time
# Third-party
import torch
# =============================================================================
# Summary: Testing vectorized mapping (vmap) to minimize memory costs
# =============================================================================
class ConstitutiveModel:
    def __init__(self):
        # Set learnable model parameters flag
        is_learnable_parameters = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set model parameters
        if is_learnable_parameters:
            # Initialize model parameters
            self._model_parameters = torch.nn.ParameterDict({})
            # Set model parameter
            self._model_parameters['param_1'] = \
                torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        else:
            # Initialize model parameters
            self._model_parameters = {}
            # Set model parameter
            self._model_parameters = {'param_1': 1.0,}
    # -------------------------------------------------------------------------
    def get_model_parameters(self):
        return self._model_parameters
# =============================================================================
class StructureMaterialState:
    def __init__(self, n_elem):
        # Initialize material models
        self._material_models = {}
        # Initialize elements material model
        self._elements_material = {str(i): None for i in range(1, n_elem + 1)}
        # Initialize elements material constitutive state variables
        self._elements_state = {}
    # -------------------------------------------------------------------------
    def init_elements_model(self, element_ids):
        # Initialize constitutive model
        constitutive_model = ConstitutiveModel()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store constitutive model
        self._material_models['1'] = constitutive_model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for element_id in element_ids:
            # Assign constitutive model
            self._elements_material[str(element_id)] = constitutive_model
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize constitutive model state variables
            self._elements_state[str(element_id)] = None
    # -------------------------------------------------------------------------
    def get_material_models(self):
        return self._material_models
    # -------------------------------------------------------------------------
    def get_elements_material(self):
        return self._elements_material
    # -------------------------------------------------------------------------     
    def update_element_state(self, elem_id, element_state):
        # Update element material constitutive state variables
        self._elements_state[str(elem_id)] = element_state
# ============================================================================= 
def compute_loss(elements_state_hist):
    # Initialize loss
    loss = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over elements
    for _, element_state_hist in elements_state_hist.items():
        # Loop over discrete time
        for element_state in element_state_hist:
            # Loop over Gauss integration points
            for _, tensor in element_state.items():
                loss += torch.sum(tensor)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss    
# =============================================================================  
class MaterialModelFinder(torch.nn.Module):
    def __init__(self):
        # Initialize from base class
        super(MaterialModelFinder, self).__init__()
    # -------------------------------------------------------------------------
    def set_specimen_data(self, specimen_material_state):
        # Set specimen material state
        self._specimen_material_state = specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect specimen underlying material models parameters
        self._set_model_parameters()
    # -------------------------------------------------------------------------
    def _set_model_parameters(self):
        # Initialize parameters
        self._model_parameters = torch.nn.ParameterDict({})
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material models
        material_models = self._specimen_material_state.get_material_models()
        # Loop over material models
        for model_key, model in material_models.items():
            # Assemble material model parameters
            if hasattr(model, 'get_model_parameters'):
                self._model_parameters[model_key] = \
                    model.get_model_parameters()
    # -------------------------------------------------------------------------
    def forward_sequential_element(self, n_elem, time_hist,
                                   elements_disp_hist):
        # Get specimen material state
        specimen_material_state = self._specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get elements material
        elements_material = specimen_material_state.get_elements_material()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements state variable history
        elements_state_hist = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for i in range(n_elem):
            # Get element label
            elem_id = i + 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get element material model
            element_material = elements_material[str(elem_id)]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get element displacement history
            element_disp_hist = elements_disp_hist[str(elem_id)]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute element state variables history
            element_state_hist = self.compute_state_variable_hist(
                element_material, element_disp_hist, time_hist)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update element material constitutive state variables
            specimen_material_state.update_element_state(
                elem_id, element_state_hist[-1])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store element state variable history
            elements_state_hist[str(elem_id)] = element_state_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute loss
        loss = compute_loss(elements_state_hist)
        # Display loss
        print(f'> Loss: {loss}')
        # Compute loss gradient
        loss.backward()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element material constitutive model parameters
        model_parameters = elements_material['1'].get_model_parameters()
        # Get material model parameter
        param_1 = model_parameters['param_1']
        # Display loss gradient w.r.t material model parameter
        print(f'> Loss gradient: {param_1.grad}')
    # -------------------------------------------------------------------------
    def compute_state_variable_hist(self, element_material, element_disp_hist,
                                    time_hist):
        # Set element number of Gauss quadrature integration points
        n_gauss = 8
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element material constitutive model parameters
        model_parameters = element_material.get_model_parameters()
        # Get material model parameter
        param_1 = model_parameters['param_1']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize element material constitutive model state variables
        # history
        element_state_hist = [{str(key): None for key in range(1, n_gauss + 1)}
                              for _ in range(n_time)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Gauss integration points
        for i in range(n_gauss):
            # Loop over discrete time
            for time_idx in range(n_time):
                # Get previous converged state variable
                if time_idx == 0:
                    state_variable_old = torch.zeros(6)
                else:
                    state_variable_old = \
                        element_state_hist[time_idx - 1][str(i + 1)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute state variable
                state_variable = \
                    (param_1*element_disp_hist[time_idx]
                     *torch.ones(6, dtype=torch.float)) + state_variable_old
                # Store state variable
                element_state_hist[time_idx][str(i + 1)] = state_variable
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return element_state_hist
    # -------------------------------------------------------------------------
    def forward_sequential_element_vmap(self, n_elem, time_hist,
                                        elements_disp_hist):
        # Get specimen material state
        specimen_material_state = self._specimen_material_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get time history length
        n_time = time_hist.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get elements material
        elements_material = specimen_material_state.get_elements_material()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # WHAT IF I HAVE DIFFERENT MATERIAL MODELS?
        element_material = elements_material['1']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # BUILD ELEMENT BATCHED INPUT TENSOR
        # (shape: n_elem x n_time)
        # Initialize tensor of elements displacement history
        elements_input_data_tensor = torch.zeros((n_elem, n_time))
        # Loop over elements
        for i in range(n_elem):
            # Get element label
            elem_id = i + 1
            # Assemble element displacement history
            elements_input_data_tensor[i, :] = elements_disp_hist[str(elem_id)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ELEMENT FUNCTION TO BE VECTORIZED
        # input shape: n_time
        # output shape: n_gp x n_time x n_features
        def func_elem(element_input_data):
            # Set element number of Gauss quadrature integration points
            n_gauss = 8
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get time history length
            n_time = element_input_data.shape[0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get element material constitutive model parameters
            model_parameters = element_material.get_model_parameters()
            # Get material model parameter
            param_1 = model_parameters['param_1']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # BUILD GAUSS POINT BATCHED INPUT TENSOR
            # (shape: n_gp x n_time)
            gausss_input_data_vlist = \
                [element_input_data for i in range(n_gauss)]
            gausss_input_data_tensor = \
                torch.stack(gausss_input_data_vlist, dim=0)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # GAUSS POINT FUNCTION TO BE VECTORIZED
            # input shape: n_time
            # output shape: n_time x n_features
            def func_gp(gauss_input_data):
                # Initialize Gauss point data
                gauss_output_data_vlist = []
                # Loop over time
                for t in range(n_time):
                    # Get previous converged state variable
                    if t == 0:
                        state_variable_old = torch.zeros(6)
                    else:
                        state_variable_old = gauss_output_data_vlist[-1]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute state variable
                    state_variable = state_variable_old \
                        + param_1*gauss_input_data[t]*torch.ones(6)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Store state variable
                    gauss_output_data_vlist.append(state_variable)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # BUILD GAUSS POINT OUTPUT TENSOR
                # (shape: n_time x n_features)
                gauss_data = torch.stack(gauss_output_data_vlist, dim=0)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                return gauss_data
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # VECTORIZE GAUSS POINT FUNCTION
            vmap_gp = torch.vmap(func_gp)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # BUILD ELEMENT OUTPUT TENSOR
            # (shape: n_gp x n_time x n_features)
            element_state_hist_tensor = vmap_gp(gausss_input_data_tensor)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            return element_state_hist_tensor
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # VECTORIZE ELEMENT FUNCTION
        vfunc_elem = torch.vmap(func_elem)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # COMPUTE MESH OUTPUT TENSOR
        # (shape: n_elem x n_gp x n_time x n_features)
        elements_state_hist_tensor = vfunc_elem(elements_input_data_tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elements state variable history
        elements_state_hist = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over elements
        for i in range(n_elem):
            # Get element label
            elem_id = i + 1
            # Get number of Gauss integration points
            n_gauss = elements_state_hist_tensor.shape[1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize element material constitutive model state variables
            # history
            element_state_hist = \
                [{str(key): None for key in range(1, n_gauss + 1)}
                 for _ in range(n_time)]
            # Loop over Gauss integration points
            for j in range(n_gauss):
                # Loop over discrete time
                for time_idx in range(n_time):
                    element_state_hist[time_idx][str(j + 1)] = \
                        elements_state_hist_tensor[i, j, time_idx, :]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update element material constitutive state variables
            specimen_material_state.update_element_state(
                elem_id, element_state_hist[-1])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store element state variable history
            elements_state_hist[str(elem_id)] = element_state_hist
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute loss
        loss = compute_loss(elements_state_hist)
        # Display loss
        print(f'> Loss: {loss}')
        # Compute loss gradient
        loss.backward()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get element material constitutive model parameters
        model_parameters = elements_material['1'].get_model_parameters()
        # Get material model parameter
        param_1 = model_parameters['param_1']
        # Display loss gradient w.r.t material model parameter
        print(f'> Loss gradient: {param_1.grad}')     
# =============================================================================
# Set number of elements
n_elem = 30
# Set time history length
n_time = 10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set time history
time_hist = torch.tensor([x for x in range(0, n_time)], dtype=torch.float)
# Set elements displacement history
elements_disp_hist = {}
for i in range(n_elem):
    elem_id = i + 1
    elements_disp_hist[str(elem_id)] = \
        elem_id*torch.ones(n_time, dtype=torch.float)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize structure material state
specimen_material_state = StructureMaterialState(n_elem)
# Initialize elements constitutive models and state variables
specimen_material_state.init_elements_model(
    element_ids=tuple(x for x in range(1, n_elem + 1)))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize material model finder
material_model_finder = MaterialModelFinder()
# Set specimen material state
material_model_finder.set_specimen_data(specimen_material_state)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the process ID (PID) of the current process
process = psutil.Process(os.getpid())
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize timer
start_time_sec = time.time()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set forward propagation mode
is_forward_vmap = True
# Display forward propagation mode
print(f'\nForward propagation mode: {is_forward_vmap}\n'
      + len(f'Forward propagation mode: {is_forward_vmap}')*'-')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Forward propagation
if is_forward_vmap:
    material_model_finder.forward_sequential_element_vmap(n_elem, time_hist,
                                                          elements_disp_hist)
else:
    material_model_finder.forward_sequential_element(n_elem, time_hist,
                                                     elements_disp_hist)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute execution time
exec_time_sec = time.time() - start_time_sec
# Display execution time
print(f'\n> Execution time: {exec_time_sec:.4f} s')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get process non-swapped physical memory usage
memory_usage = process.memory_info().rss
memory_usage_mb = memory_usage/(1024**2)
# Display memory usage
print(f'> Memory usage: {memory_usage_mb:.2f} MB')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()