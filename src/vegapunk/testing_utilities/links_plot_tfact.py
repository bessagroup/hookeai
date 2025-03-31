# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from simulators.links.links import LinksSimulator
from ioput.plots import save_figure, plot_xy_data
# =============================================================================
# Summary: Generate total load factor history plot from Links input data file
# =============================================================================
def plot_links_tfact_hist(links_input_file_path, is_stdout_display=False,
                          is_save_fig=False):
    """Plot Links total loading factor history.
    
    Parameters
    ----------
    links_input_file_path : str
        Links input data file path.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_save_fig : bool, default=False
        Save figure.
    """
    # Check if simulation input data file exists
    if not os.path.isfile(links_input_file_path):
        raise RuntimeError(f'Links simulation input data file has not been '
                           f'found:\n\n{links_input_file_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get simulation name
    simulation_name = \
        os.path.splitext(os.path.basename(links_input_file_path))[0]
    # Get Links simulation input data file directory
    links_input_file_dir = os.path.dirname(links_input_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links simulation input data file
    links_input_file = open(links_input_file_path, 'r')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reset file position
    links_input_file.seek(0)
    # Initialize search flags
    is_keyword_found = False
    # Search for Number_of_Increments or INCREMENTS keyword
    for line in links_input_file:
        if bool(re.search(r'Number_of_Increments\b', line, re.IGNORECASE)): 
            # Collect number of increments
            n_inc = int(line.split()[1])
            # Initialize incremental and total load factor history
            dfact_hist = np.zeros((n_inc + 1, 2))
            tfact_hist = np.zeros((n_inc + 1, 2))
            # Set discrete time history
            time_hist = \
                np.linspace(0.0, 1.0, n_inc + 1, endpoint=True, dtype=float)
            # Store discrete time history
            dfact_hist[:, 0] = time_hist
            tfact_hist[:, 0] = time_hist
            # Compute incremental load factor history
            dfact_hist[1:, 1] = (1.0/n_inc)*np.ones(n_inc)
            for inc in range(1, n_inc + 1):
                tfact_hist[inc, 1] = \
                    tfact_hist[inc - 1, 1] + dfact_hist[inc, 1]
            # Finished processing Number_of_Increments section
            break
        elif bool(re.search(r'INCREMENTS\b', line, re.IGNORECASE)):        
            # Start processing INCREMENTS section
            is_keyword_found = True
            # Collect number of increments
            n_inc = int(line.split()[1])
            # Initialize incremental and total load factor history
            dfact_hist = np.zeros((n_inc + 1, 2))
            tfact_hist = np.zeros((n_inc + 1, 2))
            # Set discrete time history
            time_hist = \
                np.linspace(0.0, 1.0, n_inc + 1, endpoint=True, dtype=float)
            # Store discrete time history
            dfact_hist[:, 0] = time_hist
            tfact_hist[:, 0] = time_hist
            # Initialize increment counter
            inc = 1
        elif is_keyword_found and (bool(re.search(r'^' + r'[*][A-Z]+', line))
                                   or line.strip() == ''):
            # Finished processing INCREMENTS section
            break
        elif is_keyword_found:
            # Get incremental load factor
            if len(line.split()) == 3:
                dfact = float(line.split()[0])
            elif len(line.split()) == 4:
                dfact = float(line.split()[1])
            else:
                raise RuntimeError(f'Expecting 3 or 4 parameters for each '
                                   f'increment loading data, but found '
                                   f'{len(line.split())}.')
            # Store incremental load factor
            dfact_hist[inc, 1] = dfact
            # Store total load factor
            tfact_hist[inc, 1] = tfact_hist[inc - 1, 1] + dfact
            # Update increment counter
            inc += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total accumulated load factor
    total_acc_tfact = np.sum(np.abs(dfact_hist[:, 1]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set total load factor history data file path
    tfact_hist_path = os.path.join(links_input_file_dir, 'tfact_hist.dat')
    # Open total load factor history data file
    tfact_hist_file = open(tfact_hist_path, 'w')
    # Build total load factor history file content
    write_lines = \
        ['\nLOAD FACTOR HISTORY' + '\n'] \
        + [f'\nNumber of loading increments = {n_inc}'] \
        + [f'\n\nTotal accumulated load factor = {total_acc_tfact}'] \
        + [f'\n\nIncremental (absolute) load factor history:'] \
        + [f'\n\ndfact_hist_abs = {list(np.abs(dfact_hist[:, 1]))}'] \
        + [f'\n\nTotal load factor history data:'] \
        + [f'\n\n{"Time":^16s}  {"Total Load Factor":^16s}'] \
        + [f'\n{tfact_hist[i, 0]:^16.8e}   {tfact_hist[i, 1]:^16.8e}'
        for i in range(n_inc + 1)]
    # Write total load factor history data file
    tfact_hist_file.writelines(write_lines)
    # Close total load factor history data file
    tfact_hist_file.close()    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot total load factor history data
    figure, _ = plot_xy_data(tfact_hist, x_lims=(0.0, 1.0),
                             x_label='Time', y_label='Total load factor',
                             is_latex=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        # Set figure name
        filename = 'total_load_factor_hist'
        # Save figure
        save_figure(figure, filename, format='pdf',
                    save_dir=links_input_file_dir)
# =============================================================================
def gen_links_loading_incrementation_file(save_dir, is_stdout_display=False,
                                          is_save_fig=False):
    """Generate Links loading incrementation file.
    
    Loading incrementation parameters must be directly set in
    LinksSimulator._get_links_loading_incrementation().
    
    Parameters
    ----------
    save_dir : str
        Directory where loading incrementation file is saved.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_save_fig : bool, default=False
        Save figure.
    """
    # Check saving directory
    if not os.path.exists(save_dir):
        raise RuntimeError('Saving directory has not been found:'
                           '\n\n{save_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Links simulator
    links_simulator = LinksSimulator(None, None, None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get loading incrementation
    increm_lines = links_simulator._get_links_loading_incrementation()
    # Append loading increment
    write_lines = increm_lines
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set loading incrementation data file path
    increm_file_path = os.path.join(os.path.normpath(save_dir),
                                    'loading_incrementation.dat')
    # Open loading incrementation data file
    increm_file = open(increm_file_path, 'w')
    # Write loading incrementation data file
    increm_file.writelines(write_lines)
    # Close loading incrementation data file
    increm_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Links total loading factor history
    plot_links_tfact_hist(increm_file_path,
                          is_stdout_display=is_stdout_display,
                          is_save_fig=is_save_fig)
# =============================================================================
if __name__ == "__main__":
    # Set computation processes
    is_plot_links_tfact_hist = True
    is_generate_links_tfact_hist = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot Links total loading factor history
    if is_plot_links_tfact_hist:
        # Set Links simulation input data file
        links_input_file_path = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'darpa_paper_examples/global/specimens/tensile_double_notched/'
             'meshes/hexa8_n760_e504/0_links_simulation/'
             'tensile_double_notched.dat')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set display and save plot flags
        is_stdout_display = True
        is_save_fig = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot Links total loading factor history
        plot_links_tfact_hist(links_input_file_path, is_stdout_display=False,
                              is_save_fig=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate Links loading incrementation file
    if is_generate_links_tfact_hist:
        # Set saving directory
        save_dir = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'darpa_paper_examples/global/specimens/tensile_double_notched/'
             'meshes/hexa8_n760_e504/0_links_simulation')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate Links loading incrementation file.
        gen_links_loading_incrementation_file(
            save_dir, is_stdout_display=False, is_save_fig=True)
