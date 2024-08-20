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
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, save_figure
# =============================================================================
# Summary: Generate plot by providing data explicitly
# =============================================================================
# Set plots directory
plots_dir = ('/home/bernardoferreira/Documents/brown/projects/'
             'darpa_project/5_global_specimens/memory_bottleneck')
# Set fixed parameters
n_time = 5
n_params = 1
# Set file name
filename = f'memory_vs_ne_for_nt_{n_time}_np_{n_params}'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set data array
data_array = np.zeros((5, 4))
data_array[:, 0] = [1, 10, 100, 1000, 10000]
data_array[:, 1] = [12.8, 16.4, 55.8, 390.3, 4184]
data_array[:, 2] = data_array[:, 0]
data_array[:, 3] = [12.2, 12.5, 13.0, 19.3, 72.6]
# Set data labels
data_labels = ('full_graph', 'memory_efficient')
# Set title
title = (f'$n_e = \Delta$ / $n_t = {n_time}$ / $n_p = {n_params}$')
# Set axes labels
x_label = 'Number of elements'
y_label = 'Memory (MB)'
# Set axes limits
x_lims = (None, None)
y_lims = (None, None)
# Set axes scale
x_scale = 'log'
y_scale = 'log'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot data
figure, axes = plot_xy_data(data_array, data_labels=data_labels,
                            x_lims=x_lims, y_lims=y_lims,
                            x_label=x_label, y_label=y_label,
                            x_scale=x_scale, y_scale=y_scale,
                            title=title, marker='o', is_latex=True)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display figure
plt.show()
# Save figure
save_figure(figure, filename, format='pdf', save_dir=plots_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Close plot
plt.close(figure)