"""Plots to assess training of Graph Neural Network model.

Functions
---------
plot_training_loss_history
    Plot model training process loss history.
plot_training_loss_and_lr_history
    Plot model training process loss and learning rate histories.
plot_loss_convergence_test
    Plot testing and training loss for different training data set sizes.
plot_kfold_cross_validation
    Plot k-fold cross-validation results.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data, plot_xy2_data, plot_xny_data, \
    grouped_bar_chart, save_figure
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def plot_training_loss_history(loss_history, loss_type=None, is_log_loss=False,
                               loss_scale='linear',
                               filename='training_loss_history',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=False):
    """Plot model training process loss history.
    
    Parameters
    ----------
    loss_history : dict
        One or more training processes loss histories, where each loss history
        (key, str) is stored as a list of epochs loss values (item, list).
        Dictionary keys are taken as labels for the corresponding training
        processes loss histories.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check loss history
    if not isinstance(loss_history, dict):
        raise RuntimeError('Loss history is not a dict.')
    elif not all([isinstance(x, list) for x in loss_history.values()]):
        raise RuntimeError('Data must be provided as a dict where each loss '
                           'history (key, str) is stored as a list[float] '
                           '(item, list).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of training processes
    n_loss_history = len(loss_history.keys())
    # Get maximum number of training epochs
    max_n_train_epochs = max([len(x) for x in loss_history.values()])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data array and data labels
    data_xy = np.full((max_n_train_epochs, 2*n_loss_history), fill_value=None)
    data_labels = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over training processes
    for i, (key, val) in enumerate(loss_history.items()):
        # Assemble loss history
        data_xy[:len(val), 2*i] = tuple([*range(0, len(val))])
        if is_log_loss:
            data_xy[:len(val), 2*i + 1] = tuple(np.log(val))
        else:
            data_xy[:len(val), 2*i + 1] = tuple(val)
        # Assemble data label
        data_labels.append(key)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, max_n_train_epochs)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Epochs'
    if loss_type is None:
        if is_log_loss:
            y_label = 'log(Loss)'
        else:
            y_label = 'Loss'
    else:
        if is_log_loss:
            y_label = f'log(Loss) ({loss_type})'
        else:
            y_label = f'Loss ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Training loss history'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = plot_xy_data(data_xy, data_labels=data_labels, x_lims=x_lims,
                             y_lims=y_lims, title=title, x_label=x_label,
                             y_label=y_label, y_scale=y_scale,
                             x_tick_format='int', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def plot_training_loss_and_lr_history(loss_history, lr_history, loss_type=None,
                                      is_log_loss=False, loss_scale='linear',
                                      lr_type=None,
                                      filename='training_loss_and_lr_history',
                                      save_dir=None, is_save_fig=False,
                                      is_stdout_display=False, is_latex=False):
    """Plot model training process loss and learning rate histories.
    
    Parameters
    ----------
    loss_history : list[float]
        Training process loss history stored as a list of training epochs
        loss values.
    lr_history : list[float]
        Training process learning rate history stored as a list of training
        epochs learning rate values.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    lr_type : str, default=None
        Learning rate scheduler type. If provided, then learning rate scheduler
        type is added to the y-axis label.    
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check loss history
    if not isinstance(loss_history, list):
        raise RuntimeError('Loss history is not a list[float].')
    # Check learning rate history
    if not isinstance(lr_history, list):
        raise RuntimeError('Learning rate history is not a list[float].')
    elif len(lr_history) != len(loss_history):
        raise RuntimeError('Number of epochs of learning rate history is not '
                           'consistent with loss history.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data arrays
    x = tuple([*range(0, len(loss_history))])
    if is_log_loss:
        data_xy1 = np.column_stack((x, tuple(np.log(loss_history))))
    else:
        data_xy1 = np.column_stack((x, tuple(loss_history)))
    data_xy2 = np.column_stack((x, tuple(lr_history)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, len(loss_history))
    y1_lims = (None, None)
    y2_lims = (None, None)
    y1_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Epochs'
    if loss_type is None:
        if is_log_loss:
            y1_label = 'log(Loss)'
        else:
            y1_label = 'Loss'
    else:
        if is_log_loss:
            y1_label = f'log(Loss) ({loss_type})'
        else:
            y1_label = f'Loss ({loss_type})'
    if lr_type is None:
        y2_label = 'Learning rate'
    else:
        y2_label = f'Learning rate ({lr_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Training loss and learning rate history'    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss and learning rate history
    figure, _ = plot_xy2_data(data_xy1, data_xy2, x_lims=x_lims,
                              y1_lims=y1_lims, y2_lims=y2_lims, title=title,
                              x_label=x_label, y1_label=y1_label,
                              y2_label=y2_label, y1_scale=y1_scale,
                              x_tick_format='int', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def plot_loss_convergence_test(testing_loss, training_loss=None,
                               loss_type=None, is_log_loss=False,
                               loss_scale='linear',
                               filename='loss_convergence_test',
                               save_dir=None, is_save_fig=False,
                               is_stdout_display=False, is_latex=False):
    """Plot testing and training loss for different training data set sizes.
    
    Parameters
    ----------
    testing_loss : numpyp.ndarray(2d)
        Testing loss data array where i-th row is associated with the i-th
        testing process for a given training data set size and the
        corresponding data is stored as folows: testing_loss[i, 0] is the size
        of the dataset used to train the model, testing_loss[i, 1:] is the
        testing loss for each trained model (e.g., different training data sets
        in k-fold cross-validation). Missing loss values should be stored as
        None.
    training_loss : numpy.ndarray(2d), default=None
        Training loss data array where i-th row is associated with the i-th
        training process for given training data set size and the corresponding
        data is stored as folows: training_loss[i, 0] is the size of the
        data set used to train the model, training_loss[i, 1:] is the training
        loss for each trained model (e.g., different training data sets in
        k-fold cross-validation). Missing loss values should be stored as None.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    is_log_loss : bool, default=False
        Applies logarithm to loss values if True, keeps original loss values
        otherwise.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='training_loss_history'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check testing loss data array
    if not isinstance(testing_loss, np.ndarray):
        raise RuntimeError('Testing loss data array is not a np.ndarray.')
    elif len(testing_loss.shape) != 2:
        raise RuntimeError('Testing loss data array is not a np.ndarray '
                           'of shape (n_training_sizes, n_testing_loss).')
    # Check training loss data array
    if training_loss is not None:
        if not isinstance(training_loss, np.ndarray):
            raise RuntimeError('Training loss data array is not a np.ndarray.')
        elif len(training_loss.shape) != 2:
            raise RuntimeError('Training loss data array is not a np.ndarray '
                               'of shape (n_training_sizes, n_training_loss).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply logarithm to loss
    if is_log_loss:
        testing_loss[:, 1:] = np.log(testing_loss[:, 1:])
        if training_loss is not None:
            training_loss[:, 1:] = np.log(training_loss[:, 1:])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
    # Set list of data array
    if training_loss is not None:
        data_xy_list = [training_loss, testing_loss]
    else:
        data_xy_list = [testing_loss,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data labels
    data_labels = None
    if training_loss is not None:
        data_labels = ['Training', 'Testing']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes limits and scale
    x_lims = (0, None)
    y_lims = (None, None)
    y_scale = loss_scale
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = 'Training data set size'
    if loss_type is None:
        if is_log_loss:
            y_label = 'log(Loss)'
        else:
            y_label = 'Loss'
    else:
        if is_log_loss:
            y_label = f'log(Loss) ({loss_type})'
        else:
            y_label = f'Loss ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'Data set size convergence test'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = plot_xny_data(data_xy_list, range_type='mean-std',
                              data_labels=data_labels, x_lims=x_lims,
                              y_lims=y_lims, title=title, x_label=x_label,
                              y_label=y_label, y_scale=y_scale,
                              x_tick_format='int', is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)
# =============================================================================
def plot_kfold_cross_validation(k_fold_loss_array, loss_type=None,
                                loss_scale='linear',
                                filename='kfold_cross_validation',
                                save_dir=None, is_save_fig=False,
                                is_stdout_display=False, is_latex=False):
    """Plot k-fold cross-validation results.
    
    Parameters
    ----------
    k_fold_loss_array : numpy.ndarray(2d)
        k-fold cross-validation loss array. For the i-th fold,
        data_array[i, 0] stores the best training loss and data_array[i, 1]
        stores the average prediction loss per sample.
    loss_type : str, default=None
        Loss type. If provided, then loss type is added to the y-axis label.
    loss_scale : {'linear', 'log'}, default='linear'
        Loss axis scale type.
    filename : str, default='prediction_vs_groundtruth'
        Figure name.
    save_dir : str, default=None
        Directory where figure is saved. If None, then figure is saved in
        current working directory.
    is_save_fig : bool, default=False
        Save figure.
    is_stdout_display : bool, default=False
        True if displaying figure to standard output device, False otherwise.
    is_latex : bool, default=False
        If True, then render all strings in LaTeX. If LaTex is not available,
        then this option is silently set to False and all input strings are
        processed to remove $(...)$ enclosure.
    """
    # Check loss history
    if not isinstance(k_fold_loss_array, np.ndarray):
        raise RuntimeError('k-fold cross-validation loss array must be '
                           'numpy.ndarray(2d) of shape (n_fold, 2).')
    elif k_fold_loss_array.shape[1] != 2:
        raise RuntimeError('k-fold cross-validation loss array must be '
                           'numpy.ndarray(2d) of shape (n_fold, 2).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get number of cross-validation folds
    n_fold = k_fold_loss_array.shape[0]
    # Set folds labels
    folds_labels = tuple([f'Fold {x + 1}' for x in range(n_fold)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build folds training and validation loss data
    folds_data = {}
    folds_data['Training'] = tuple(k_fold_loss_array[:, 0])
    folds_data['Validation'] = tuple(k_fold_loss_array[:, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axes labels
    x_label = None
    y_label = 'Loss'
    if loss_type is not None:
        y_label += f' ({loss_type})'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set title
    title = 'k-Fold Cross-Validation'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot loss history
    figure, _ = grouped_bar_chart(groups_labels=folds_labels,
                                  groups_data=folds_data,
                                  is_avg_hline=True,
                                  title=title, x_label=x_label,
                                  y_label=y_label, y_scale=loss_scale,
                                  is_latex=is_latex)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display figure
    if is_stdout_display:
        plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save figure
    if is_save_fig:
        save_figure(figure, filename, format='pdf', save_dir=save_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close plot
    plt.close(figure)