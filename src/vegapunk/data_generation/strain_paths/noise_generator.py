"""Generate noisy strain loading paths.

Classes
-------
NoiseGenerator
    Noise generator.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class NoiseGenerator:
    """Noise generator.
    
    Attributes
    ----------
    _noise_distribution : str, {'uniform', 'gaussian', 'spiked_gaussian'}
        Noise distribution type.
    _noise_parameters : dict
        Noise distribution parameters.

    Methods
    -------
    set_noise_distribution(self, noise_distribution)
        Set noise distribution type.
    set_noise_parameters(self, noise_parameters)
        Set noise distribution parameters.
    get_required_parameters(cls, noise_distribution)
        Get required parameters for given noise distribution type.
    generate_noise_path(self, noiseless_path, \
                        noise_variability='homoscedastic', \
                        heteroscedastic_weights=None)
        Generate noise path.
    """
    def __init__(self):
        """Constructor."""
        # Initialize noise distribution
        self._noise_distribution = None
        # Initialize noise distribution parameters
        self._noise_parameters = None
    # -------------------------------------------------------------------------
    def set_noise_distribution(self, noise_distribution):
        """Set noise distribution type.
        
        Parameters
        ----------
        noise_distribution : str, {'uniform', 'gaussian', 'spiked_gaussian'}
            Noise distribution type.
        """
        # Check noise distribution type
        if noise_distribution not in ('uniform', 'gaussian',
                                      'spiked_gaussian'):
            raise RuntimeError('Unknown noise distribution type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set noise distribution type
        self._noise_distribution = noise_distribution
    # -------------------------------------------------------------------------
    def set_noise_parameters(self, noise_parameters):
        """Set noise distribution parameters.
        
        Parameters
        ----------
        noise_parameters : dict
            Noise distribution parameters.
        """
        # Get required parameters
        required_parameters = \
            self.get_required_parameters(self._noise_distribution)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check parameters
        for parameter in required_parameters:
            if parameter not in noise_parameters.keys():
                raise RuntimeError(f'Parameter {parameter} must be provided '
                                   f'for noise distribution of type '
                                   f'{self._noise_distribution}.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set parameters
        self._noise_parameters = noise_parameters  
    # -------------------------------------------------------------------------
    @classmethod
    def get_required_parameters(cls, noise_distribution):
        """Get required parameters for given noise distribution type.
        
        Parameters
        ----------
        noise_distribution : str, {'uniform', 'gaussian'}
            Noise distribution type.

        Returns
        -------
        required_noise_parameters : tuple[str]
            Noise distribution required parameters.
        """
        if noise_distribution == 'uniform':
            required_parameters = ('amp',)
        elif noise_distribution == 'gaussian':
            required_parameters = ('std',)
        elif noise_distribution == 'spiked_gaussian':
            required_parameters = ('std', 'spike', 'p_spike')
        else:
            raise RuntimeError('Unknown noise distribution type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return required_parameters
    # -------------------------------------------------------------------------
    def generate_noise_path(self, noiseless_path,
                            noise_variability='homoscedastic',
                            heteroscedastic_weights=None):
        """Generate noise path.
        
        Noise is applied independently for each signal feature.
        
        Parameters
        ----------
        noiseless_path : numpy.ndarray(2d)
            Noiseless signal path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_features). 
        noise_variability: str, {'homoscedastic', 'heteroscedastic'}, \
                           default='homoscedastic'
            Variability of noise across the data. In 'homoscedastic' noise, the
            variance of the noise remains constant across the data points
            (uniform effect regardless of independent variable). In
            'heteroscedastic' noise, the variance of the noise depends on the
            data point.
        heteroscedastic_weights : numpy.ndarray(1d), default=None
            Weights that materialize noise heteroscedasticity by scaling the
            noise distribution variance for each data point. Stored as
            numpy.ndarray(1d) of shape (sequence_length). If None, then
            defaults to ones (homoscedastic noise).

        Returns
        -------
        noise_path : numpy.ndarray(2d)
            Noise path history stored as numpy.ndarray(2d) of shape
            (sequence_length, n_features).
        """
        # Set noise path shape
        noise_path_shape = noiseless_path.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate noise path
        if noise_variability == 'homoscedastic':
            # Sample noise path
            if self._noise_distribution == 'uniform':
                # Set bounds
                low = -0.5*abs(self._noise_parameters['amp'])
                high = -low
                # Sample noise
                noise_path = np.random.uniform(low, high,
                                               size=noise_path_shape)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif self._noise_distribution in ('gaussian', 'spiked_gaussian'):
                # Set standard deviation
                std = self._noise_parameters['std']
                # Sample noise
                noise_path = np.random.normal(loc=0.0, scale=std,
                                              size=noise_path_shape)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif noise_variability == 'heteroscedastic':
            # Set heteroscedasticity weights
            if heteroscedastic_weights is None:
                heteroscedastic_weights = np.ones(noise_path_shape[0])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize noise path
            noise_path = np.zeros(noise_path_shape)
            # Loop over time steps
            for t in range(noise_path.shape[0]):
                # Get heteroscedasticity weight
                weight = heteroscedastic_weights[t]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Sample noise
                if self._noise_distribution == 'uniform':
                    # Set homoscedastic bounds
                    hom_low = -0.5*abs(self._noise_parameters['amp'])
                    hom_high = -hom_low
                    # Set heteroscedastic bounds
                    het_low = -0.5*abs(self._noise_parameters['amp'])*weight
                    het_high = -het_low
                    # Sample noise
                    noise = (np.random.uniform(hom_low, hom_high,
                                               size=noise_path_shape[1])
                             + np.random.uniform(het_low, het_high,
                                                 size=noise_path_shape[1]))
                    
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif self._noise_distribution in ('gaussian',
                                                  'spiked_gaussian'):
                    # Set homoscedastic standard deviation
                    hom_std = self._noise_parameters['std']
                    # Set heteroscedastic standard deviation
                    het_std = self._noise_parameters['std']*weight
                    # Sample noise
                    noise = (np.random.normal(loc=0.0, scale=hom_std,
                                              size=noise_path_shape[1])
                             + np.random.normal(loc=0.0, scale=het_std,
                                                size=noise_path_shape[1]))
                    
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble noise
                noise_path[t, :] = noise
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add noise spike
        if self._noise_distribution in ('spiked_gaussian',):
            # Set spike magnitude and probability
            spike = self._noise_parameters['spike']
            p_spike = self._noise_parameters['p_spike']
            # Sample noise spike
            spike_path = spike*np.random.binomial(n=1, p=p_spike,
                                                  size=noise_path_shape)
            # Add noise spike to noise path
            noise_path += spike_path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return noise_path