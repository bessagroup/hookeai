"""Data denoising methods.

Classes
-------
Denoiser
    Data denoiser.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
# Local
from ioput.plots import plot_xy_data
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class Denoiser:
    """Data denoiser.

    Methods
    -------
    denoise(self, tensor, denoise_method, denoise_parameters={})
        Denoise features tensor.
    _dn_moving_average(self, tensor, window_size)
        Denoising method: Moving Average.
    _dn_savitzky_golay(self, tensor, window_size, poly_order)
        Denoising method: Savitzky-Golay filter.
    _dn_frequency_low_pass(self, tensor, cutoff_frequency, \
                           is_plot_magnitude_spectrum=False)
        Denoising method: Frequency low-pass filter.
    _check_tensor(self, tensor)
        Check features tensor to be denoised.
    """
    def __init__(self):
        """Constructor."""
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def denoise(self, tensor, denoise_method, denoise_parameters={}):
        """Denoise features tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        denoise_method : str
            Denoising method.
        denoise_parameters : dict, default={}
            Denoising method parameters.

        Returns
        -------
        dn_tensor : torch.Tensor
            Denoised features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        """
        # Check features tensor
        self._check_tensor(tensor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize denoising method
        if denoise_method == 'moving_average':
            denoiser = self._dn_moving_average
        elif denoise_method == 'savitzky_golay':
            denoiser = self._dn_savitzky_golay
        elif denoise_method == 'frequency_low_pass':
            denoiser = self._dn_frequency_low_pass
        elif denoise_method == 'kalman_filter':
            denoiser = self._dn_kalman_filter
        elif denoise_method == 'gaussian_filter':
            denoiser = self._dn_gaussian_filter
        else:
            raise RuntimeError('Unknown denoising method.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Denoise features tensor
        dn_tensor = denoiser(tensor, **denoise_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dn_tensor
    # -------------------------------------------------------------------------
    def _dn_moving_average(self, tensor, window_size):
        """Denoising method: Moving Average.
        
        Convolution is performed independently for each feature.
        
        Edge padding is performed to preserve input tensor shape.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        window_size : int
            Convolution kernel size.

        Returns
        -------
        dn_tensor : torch.Tensor
            Denoised features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        """
        # Get number of features
        n_features = tensor.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape to (batch_size=1, n_features, sequence_length)
        tensor = tensor.T.unsqueeze(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute required padding to preserve input tensor shape
        padding = (window_size//2, window_size - 1 - window_size//2)
        # Perform edge padding (replicate edge values of each feature)
        tensor = F.pad(tensor, (padding[0], padding[1]), mode='replicate')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set convolution kernel of shape (n_features, 1, window_size)
        # (set independent kernel for each feature)
        kernel = torch.ones(n_features, 1, window_size)/window_size
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute moving average (independent convolution of each feature)
        denoised_tensor = \
            F.conv1d(tensor, kernel, padding=0, groups=n_features)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape back to (sequence_length, n_features)
        dn_tensor = denoised_tensor.squeeze(0).T
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dn_tensor
    # -------------------------------------------------------------------------
    def _dn_savitzky_golay(self, tensor, window_size, poly_order):
        """Denoising method: Savitzky-Golay filter.
        
        Convolution is performed independently for each feature.
        
        Edge padding is performed to preserve input tensor shape.
        
        General thumb rules:
        
        1. Set an odd number for the window_size (ensures that there is a
           central point around which the smoothing occurs)
        
        2. The window_size should be around 5% to 15% of the sequence length,
           depending on the level of noise. Larger window_size smooths more
           effectively but may remove important details, while a smaller
           window preserves more detail but may not adequately smooth out noise
           
        3. The poly_order must be less than window_size (ensures that there are
           enough points to fit the polynomial correctly)
           
        4. The poly_order is usually between 1 and 4. Lower poly_order is
           effective when data has a simple trend or is relative smooth, while
           larger poly_order is effective for data that is inherently more
           complex. Higher polynomial orders may introduce undesired
           oscillations!
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        window_size : int
            Convolution kernel size (number of Savitzky-Golay coefficients).
        poly_order : int
            Order of polynomial used to fit the samples in the window. Must
            be less that window_size.

        Returns
        -------
        dn_tensor : torch.Tensor
            Denoised features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        """
        # Get number of features
        n_features = tensor.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape to (batch_size=1, n_features, sequence_length)
        tensor = tensor.T.unsqueeze(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute required padding to preserve input tensor shape
        padding = (window_size//2, window_size - 1 - window_size//2)
        # Perform edge padding (replicate edge values of each feature)
        tensor = F.pad(tensor, (padding[0], padding[1]), mode='replicate')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute the coefficients for 1D Savitzky-Golay filter
        coeffs = np.array([scipy.signal.savgol_coeffs(window_size, poly_order)
                           for _ in range(n_features)])
        # Set convolution kernel of shape (n_features, 1, window_size)
        # (set independent kernel for each feature)
        kernel = torch.tensor(coeffs, dtype=torch.float).view(
            n_features, 1, window_size)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute moving average (independent convolution of each feature)
        denoised_tensor = \
            F.conv1d(tensor, kernel, padding=0, groups=n_features)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reshape back to (sequence_length, n_features)
        dn_tensor = denoised_tensor.squeeze(0).T
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dn_tensor
    # -------------------------------------------------------------------------
    def _dn_frequency_low_pass(self, tensor, cutoff_frequency,
                               is_plot_magnitude_spectrum=False):
        """Denoising method: Frequency low-pass filter.
        
        Filtering is performed independently for each feature.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        cutoff_frequency : int
            Cutoff frequency of the low-pass filter.
        is_plot_magnitude_spectrum : bool, default=False
            If True, then plot magnitude spectrum and cutoff frequency.

        Returns
        -------
        dn_tensor : torch.Tensor
            Denoised features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        """
        # Get sequence length
        n_time = tensor.shape[0]
        # Get number of features
        n_features = tensor.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform Discrete Fourier Transform
        # (independently for each feature)
        tensor_dft = torch.fft.fft(tensor, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set sampling period
        sampling_period = 1
        # Get discrete frequencies
        discrete_freqs = torch.fft.fftfreq(n_time, sampling_period)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot magnitude spectrum and cutoff frequency
        if is_plot_magnitude_spectrum:
            # Compute magnitude spectrum
            magnitude = np.abs(tensor_dft)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize data array
            data_xy = torch.zeros((n_time//2, n_time//2))
            # Build data array
            for i in range(n_features):
                data_xy[:, 2*i] = discrete_freqs[:n_time//2]
                data_xy[:, 2*i + 1] = magnitude[:n_time//2, i]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot only the first feature
            data_xy = data_xy[:, (0, 1)]
            # Set data labels
            data_labels = ('Feature 1',)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set axes limits (limited by Nyquist frequency)
            x_lims = (0, (sampling_period**-1)/2)
            y_lims = (None, None)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set axes labels
            x_label = 'Frequency (Hz)'
            y_label = 'Magnitude'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot parameter history
            _, axes = plot_xy_data(data_xy, data_labels=data_labels,
                                   x_lims=x_lims, y_lims=y_lims,
                                   x_label=x_label, y_label=y_label,
                                   marker='o', is_latex=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot cutoff frequency
            axes.axvline(cutoff_frequency, color='red', linestyle='--',
                         label='Cutoff frequency')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display legend
            axes.legend()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Display figure
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create low-pass filter
        low_pass_filter = torch.abs(discrete_freqs) <= cutoff_frequency
        # Expand low-pass filter for all features
        low_pass_filter = \
            low_pass_filter.view(-1, 1).expand(n_time, n_features)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Apply low-pass filter
        tensor_dft = low_pass_filter*tensor_dft
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform Inverse Discrete Fourier Transform
        # (independently for each feature)
        dn_tensor = torch.fft.ifft(tensor_dft, dim=0).real
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return dn_tensor
    # -------------------------------------------------------------------------
    def _check_tensor(self, tensor):
        """Check features tensor to be denoised.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Features PyTorch tensor stored as torch.Tensor with shape
            (sequence_length, n_features).
        """
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError('Features tensor is not a torch.Tensor.')
        elif len(tensor.shape) != 2:
            raise RuntimeError('Features tensor is not a torch.Tensor with '
                               'shape (sequence_length, n_features).')
    
