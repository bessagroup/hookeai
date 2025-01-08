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
# Set small value
delta = 1e-10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set number of pruning steps
n_prun_step = 10
# Set minimum data set size ratio
min_size_ratio = 0.05
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute number of pruned samples
n_prun_sample = \
    int(np.floor((1.0 - min_size_ratio)*(n_full/n_prun_step) + delta))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display
print(f'\n> Full data set size: {n_full}')
print(f'\n> Number of pruning steps: {n_prun_step}')
print(f'\n> Number of training procedures: {2*n_prun_step + 1}')
print(f'\n> Number of testing procedures: {n_prun_step}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display
print(f'\n> Data set pruning iterations:')
print(f'\n{"iteration":>11s} '
      f'{"development":>14s} {"%full":>6s} '
      f'{"training":>11s} {"%full":>6s} '
      f'{"validation":>13s} {"%full":>6s} '
      f'{"unused":>9s} {"%full":>6s}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute number of iterations
n_iter = n_prun_step + 1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize plot data
data_xy = np.zeros((n_iter, 6))
# Set data labels
data_labels = ('Development', 'Training', 'Validation')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over iterations
for iter in range(n_iter):
    # Compute development data set size (training + validation)
    n_dev = n_full - iter*n_prun_sample
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
    # Display
    print(f'{iter:>11d} '
          f'{n_dev:>14d} {ratio_dev:>6.1f} '
          f'{n_train:>11d} {ratio_train:>6.1f} '
          f'{n_valid:>13d} {ratio_valid:>6.1f} '
          f'{n_unused:>9d} {ratio_unused:>6.1f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append plot data
    data_xy[iter, :] = \
        [iter, ratio_dev, iter, ratio_train, iter, ratio_valid]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot data set iterations
figure, _ = plot_xy_data(data_xy,
                         data_labels=data_labels,
                         x_label='Iterations',
                         y_label='\% Full data set',
                         marker='o',
                         is_latex=True)
# Display plot
plt.show()