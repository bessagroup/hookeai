import torch
import matplotlib.pyplot as plt
# =============================================================================
# Summary: Simple benchmark on how to optimize parametric model with PyTorch
# =============================================================================
# Create ground-truth model
def ground_truth(x):
    y = 2.0*x + 1.0
    return y
# =============================================================================
# Set training dataset size
dataset_size = 10
# Generate training dataset
dataset = []
for i in range(dataset_size):
    input = torch.rand(1)
    target = ground_truth(input)
    dataset.append((input, target))
# =============================================================================
# Set standard parametric model
class StandardParametricModel:
    def __init__(self):
        self.a = torch.zeros(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.parameters = (self.a, self.b)
    # -------------------------------------------------------------------------
    def forward(self, input):
        output = self.a*input + self.b
        return output
    # -------------------------------------------------------------------------
    def get_model_parameters_str(self):
        model_parameters_str = f'({float(self.a)}, {float(self.b)})'
        return model_parameters_str
# =============================================================================
# Set PyTorch model
class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._parameter_option = 'saved_in_dict'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self._parameter_option == 'saved_as_attr':
            self.a = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.b = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        elif self._parameter_option == 'saved_in_dict':
            self._model_parameters = torch.nn.ParameterDict({})
            self._model_parameters['a'] = \
                torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self._model_parameters['b'] = \
                torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self._model_parameters['nested_dict'] = torch.nn.ParameterDict({})
            self._model_parameters['nested_dict']['c'] =  \
                torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self._model_parameters['nested_dict']['d'] =  0.0
    # -------------------------------------------------------------------------
    def forward(self, input):
        if self._parameter_option == 'saved_as_attr':
            output = self.a*input + self.b
        elif self._parameter_option == 'saved_in_dict':
            output = (self._model_parameters['a']*input
                      + self._model_parameters['b'])
        return output
    # -------------------------------------------------------------------------
    def get_model_parameters_str(self):
        if self._parameter_option == 'saved_as_attr':
            model_parameters = {'a': self.a, 'b': self.b}
        elif self._parameter_option == 'saved_in_dict':
            model_parameters = self._model_parameters
        model_parameters_str = \
            f'({float(model_parameters["a"])}, {float(model_parameters["b"])})'
        return model_parameters_str
# =============================================================================
# Set model type
model_type = 'torch_module'
# Initialize model
if model_type == 'standard_parametric':
    # Initialize model
    model = StandardParametricModel()
    # Get model parameters
    model_parameters = model.parameters
elif model_type == 'torch_module':
    # Initialize model
    model = PyTorchModel()
    # Get model parameters
    model_parameters = model.parameters(recurse=True)
# =============================================================================
# Set loss function
loss_function = torch.nn.MSELoss()
# Set optimizer
optimizer = torch.optim.Adam(model_parameters, lr=1e-2)
# =============================================================================
# Set number of epochs
n_epochs = 100
# Initialize number of training steps
step = 0
# Loop over epochs
for epoch in range(n_epochs):
    # Loop over samples
    for sample in dataset:
        # Get sample
        input, target = sample
        # Get model prediction
        prediction = model.forward(input)
        # Compute loss
        loss = loss_function(prediction, target)
        # Initialize gradients (set to zero)
        optimizer.zero_grad()
        # Compute gradients (backpropagation)
        loss.backward()
        # Perform optimization step
        optimizer.step()
        # Increment training step counter
        step += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display iteration results
        if True:
            print(f'\nTraining step: {step}')
            for name, param in model.named_parameters():
                print(f'\n  > Parameter ({name}): {param.data}')
                print(f'  > Gradient (wtr to {name}) ', param.grad)
            print('\n' + 80*'-')
# =============================================================================
# Display results
print('\nFinal results:')
print(f'\n  > Ground-truth = ({2.0}, {1.0})')
print(f'  > Optimization = ' + model.get_model_parameters_str())
# =============================================================================
# Collect training dataset inputs, targets and predictions
inputs = [sample[0] for sample in dataset]
targets = [sample[1] for sample in dataset]
predictions = [model.forward(sample[0]).detach().numpy() for sample in dataset]
# Plot model predictions
figure, axes = plt.subplots()
axes.set(xlabel='x', ylabel='y')
axes.plot(inputs, targets, label='Ground-truth')
axes.plot(inputs, predictions, label='Prediction')
axes.legend()
plt.show()