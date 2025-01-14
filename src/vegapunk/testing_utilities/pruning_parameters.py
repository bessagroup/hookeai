# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Third-party
import numpy as np
# Local
from ioput.plots import plot_xy_data
import matplotlib.pyplot as plt
# =============================================================================
# Summary: Assess data set pruning parameters and iterations
# =============================================================================
# Display
string = 'Data set pruning'
sep = len(string)*'-'
print(f'\n{string}\n{sep}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set full data set size
n_full = 1000
# Set training and validation split sizes
split_sizes = {'training': 0.8, 'validation': 0.2}
# Set pruning training, validation and testing split sizes
pruning_split_sizes = {'training': 0.7, 'validation': 0.2, 'testing': 0.1}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set number of pruned samples
n_prun_sample = 10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set pruning termination criteria parameters:
# Set minimum development data set size
n_dev_min = 10
# Set minimum pruning testing data set size
n_prun_test_min = 10
# Set maximum pruned samples testing ratio
prun_test_ratio_max = 0.5
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display
print(f'\n> Full data set size: {n_full}')
print(f'\n> Number of pruned samples (per pruning step): {n_prun_sample}')
print(f'\n> Minimum development data set size: {n_dev_min}')
print(f'\n> Minimum pruning testing data set size: {n_prun_test_min}')
print(f'\n> Maximum pruned samples testing ratio: {prun_test_ratio_max}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display
print(f'\n> Data set pruning iterations:')
print(f'\n{"iteration":>11s} '
      f'{"development":>14s} {"%full":>6s} '
      f'{"training":>11s} {"%full":>6s} '
      f'{"validation":>13s} {"%full":>6s} '
      f'{"unused":>9s} {"%full":>6s}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize pruning iterative data
pruning_iter_data = []
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize pruning iteration counter
iter = 0
# Initialize pruning termination flag
is_keep_pruning = True
# Loop over iterations
while is_keep_pruning:
    # Compute development data set size (training + validation)
    if iter == 0:
        n_dev = n_full
    else:
        n_dev = n_dev - n_prun_sample
    # Compute ratio
    ratio_dev = (n_dev/n_full)*100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute pruning training data set size
    n_prun_train = int(np.floor(pruning_split_sizes['training']*n_dev))
    # Compute pruning validation data set size
    n_prun_valid = int(np.floor(pruning_split_sizes['validation']*n_dev))
    # Compute pruning testing data set size
    n_prun_test = int(np.floor(pruning_split_sizes['testing']*n_dev))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check pruning termination criteria
    termination_status = None
    if n_dev <= n_dev_min:
        termination_status = 'Minimum development data set size reached.'
    elif n_prun_test < n_prun_test_min:
        termination_status = 'Minimum pruning testing data set size reached.'
    elif n_prun_sample/n_prun_test > prun_test_ratio_max:
        termination_status = 'Maximum pruned samples testing ratio reached.'
    # Terminate pruning
    if termination_status is not None:
        # Display termination status
        print(f'\n> Termination status: {termination_status}')
        # Terminate pruning
        is_keep_pruning = False
        continue
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display pruning step
    if iter > 0:
        print(35*' '
              + f'(pruning step: '
              f'T{n_prun_train}|V{n_prun_valid}|T{n_prun_test})')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute training data set size
    n_train = int(np.floor(split_sizes['training']*n_dev))
    ratio_train = (n_train/n_full)*100
    # Compute validation data set size
    n_valid = n_dev - n_train
    ratio_valid = (n_valid/n_full)*100
    # Compute unused data set size
    n_unused = iter*n_prun_sample
    ratio_unused = (n_unused/n_full)*100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display pruning iteration
    print(f'{iter:>11d} '
          f'{n_dev:>14d} {ratio_dev:>6.1f} '
          f'{n_train:>11d} {ratio_train:>6.1f} '
          f'{n_valid:>13d} {ratio_valid:>6.1f} '
          f'{n_unused:>9d} {ratio_unused:>6.1f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store pruning iterative data
    pruning_iter_data.append(
        [iter, ratio_dev, iter, ratio_train, iter, ratio_valid])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update pruning iteration counter
    iter += 1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get number of pruning iterations
n_iter = len(pruning_iter_data)
# Initialize plot data
data_xy = np.zeros((n_iter, 6))
# Assemble plot data
for iter in range(n_iter):
    data_xy[iter, :] = pruning_iter_data[iter]
# Set data labels
data_labels = ('Development', 'Training', 'Validation')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot data set iterations
figure, _ = plot_xy_data(data_xy,
                         data_labels=data_labels,
                         x_label='Iterations',
                         y_label='\% Full data set',
                         marker='o',
                         is_latex=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display plot
plt.show()