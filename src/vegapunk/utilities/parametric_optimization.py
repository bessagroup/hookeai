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
# Set model
class Model:
    def __init__(self):
        self.a = torch.zeros(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.parameters = (self.a, self.b)
    def forward(self, input):
        output = self.a*input + self.b
        return output
# =============================================================================
# Initialize model
model = Model()
# =============================================================================
# Set loss function
loss_function = torch.nn.MSELoss()
# Set optimizer
optimizer = torch.optim.Adam(model.parameters, lr=1e-2)
# =============================================================================
# Set number of epochs
n_epochs = 100
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
# =============================================================================
# Display results
print(f'Ground-truth = ({2.0}, {1.0})')
print(f'Optimization = ({model.a[0]}, {model.b[0]})')
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