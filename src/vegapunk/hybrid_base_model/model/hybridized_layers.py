"""Hybridization layers for hybrid model (HybridModel).

Classes
-------
BatchedElasticModel(torch.nn.Module)
    Linear elastic constitutive model for batched time series data.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import torch
# Local
from simulators.fetorch.material.models.standard.elastic import Elastic
from simulators.fetorch.math.matrixops import get_problem_type_parameters, \
    vget_state_2Dmf_from_3Dmf
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class BatchedElasticModel(torch.nn.Module):
    """Linear elastic constitutive model for batched time series data.
    
    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _elastic_properties : dict
        Elastic material properties (key, str) values (item, float).
        Expecting independent elastic moduli ('Eijkl') according to
        elastic symmetries. Young modulus ('E') and Poisson's ratio ('v')
        may alternatively be provided under elastic isotropy.
    _elastic_symmetry : {'isotropic', 'transverse_isotropic', \
                         'orthotropic', 'monoclinic', 'triclinic'}, \
                        default='isotropic'
        Elastic symmetries:

        * 'triclinic':  assumes no elastic symmetries.

        * 'monoclinic': assumes plane of symmetry 12.

        * 'orthotropic': assumes planes of symmetry 12 and 13.

        * 'transverse_isotropic': assumes axis of symmetry 3

        * 'isotropic': assumes complete symmetry.

    _comp_order_sym : tuple[str]
        Strain/Stress components symmetric order.
    _n_strain_comp : int
        Number of strain/stress components.
    _elastic_tangent_mf : torch.Tensor(2d)
        Elasticity tensor (matricial form).
    _kelvin_matrix : torch.Tensor(1d)
        Kelvin factors transformation matrix associated with strain components
        symmetric order.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, features_in)
        Forward propagation.
    """
    def __init__(self, problem_type, elastic_properties, elastic_symmetry,
                 device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        elastic_properties : dict
            Elastic material properties (key, str) values (item, float).
            Expecting independent elastic moduli ('Eijkl') according to
            elastic symmetries. Young modulus ('E') and Poisson's ratio ('v')
            may alternatively be provided under elastic isotropy.
        elastic_symmetry : {'isotropic', 'transverse_isotropic', \
                            'orthotropic', 'monoclinic', 'triclinic'}, \
                            default='isotropic'
            Elastic symmetries:

            * 'triclinic':  assumes no elastic symmetries.

            * 'monoclinic': assumes plane of symmetry 12.

            * 'orthotropic': assumes planes of symmetry 12 and 13.

            * 'transverse_isotropic': assumes axis of symmetry 3

            * 'isotropic': assumes complete symmetry.

        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(BatchedElasticModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set problem type
        self._problem_type = problem_type
        # Set elastic material properties
        self._elastic_properties = elastic_properties
        # Set elastic symmetry
        self._elastic_symmetry = elastic_symmetry
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        self._n_dim, self._comp_order_sym, _ = \
            get_problem_type_parameters(problem_type)
        # Set number of strain components
        self._n_strain_comp = len(self._comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute 3D elasticity tensor
        self._elastic_tangent_mf = Elastic.elastic_tangent_modulus(
            elastic_properties, elastic_symmetry='isotropic',
            device=self._device)
        # Extract 2D elasticity tensor
        if self._n_dim == 2:
            self._elastic_tangent_mf = vget_state_2Dmf_from_3Dmf(
                self._elastic_tangent_mf, device=self._device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Kelvin factors transformation matrix
        self._kelvin_matrix = \
            torch.tensor([math.sqrt(2) if x[0] != x[1] else 1.0
                          for x in self._comp_order_sym],
                         device=self._device)
    # -------------------------------------------------------------------------
    def set_device(self, device_type):
        """Set device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        if device_type in ('cpu', 'cuda'):
            if device_type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PyTorch with CUDA is not available. '
                                   'Please set the model device type as CPU '
                                   'as:\n\n' + 'model.set_device(\'cpu\').')
            self._device_type = device_type
            self._device = torch.device(device_type)
        else:
            raise RuntimeError('Invalid device type.')
    # -------------------------------------------------------------------------
    def get_device(self):
        """Get device on which torch.Tensor is allocated.
        
        Returns
        -------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device
    # -------------------------------------------------------------------------
    def forward(self, features_in):
        """Forward propagation.
        
        The tensor of input features can be one of two (infered from number of
        features): (1) the elastic strain or (2) the concatenated plastic and
        total strain. In the second case, the elastic strain is computed from
        the elastoplastic additive decomposition of the total strain.
        
        The tensor of output features is the stress.

        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Check input state features
        if not isinstance(features_in, torch.Tensor):
            raise RuntimeError('Input features were not provided as '
                               'torch.Tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if batched input
        is_batched = features_in.dim() == 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batching dimension
        batch_dim = 1
        # Set batched dimension
        if not is_batched:
            features_in = features_in.unsqueeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic strain
        if features_in.shape[-1] == self._n_strain_comp:
            # Input features are already the elastic strain
            pass
        elif features_in.shape[-1] == 2*self._n_strain_comp:
            # Input features are the concatenated total and plastic strains
            features_in = (features_in[:, :, self._n_strain_comp:]
                           - features_in[:, :, :self._n_strain_comp])
        else:
            raise RuntimeError(f'Number of input features '
                               f'({features_in.shape[-1]}) must be equal to '
                               f'the number of strain components (elastic '
                               f'strain) or equal to twice the number of '
                               f'strain components (concatenated total and '
                               f'plastic strain), {self._n_strain_comp} or '
                               f'{2*self._n_strain_comp}, respectively.')                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input shape
        sequence_length, batch_size, n_features_in = features_in.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate features sequences
        # (shape: sequence_length*batch_size x n_features_in)
        features_in = features_in.view(-1, n_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute output features
        features_out = ((1.0/self._kelvin_matrix.view(-1, 1))*torch.matmul(
            self._elastic_tangent_mf,
            self._kelvin_matrix.view(-1, 1)*(features_in.T))).T
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of output features
        n_features_out = features_out.shape[-1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert features sequences concatenation
        # (shape: sequence_length x batch_size x n_features_out)
        features_out = features_out.view(
            sequence_length, batch_size, n_features_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            features_out = features_out.squeeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out
# =============================================================================
class VolDevCompositionModel(torch.nn.Module):
    """Volumetric/Deviatoric composition model for batched time series data.
    
    Performs volumetric/deviatoric composition of a second-order strain or
    stress tensor, i.e., builds tensor from the associated volumetric and
    deviatoric componets.
    
    Attributes
    ----------
    _comp_order_sym : tuple[str]
        Strain/Stress components symmetric order.
    _n_comp : int
        Number of strain/stress components.
    _device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    _device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    set_device(self, device_type)
        Set device on which torch.Tensor is allocated.
    get_device(self)
        Get device on which torch.Tensor is allocated.
    forward(self, features_in)
        Forward propagation.
    """
    def __init__(self, device_type='cpu'):
        """Constructor.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}, default='cpu'
            Type of device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(VolDevCompositionModel, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        self.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        _, self._comp_order_sym, _ = get_problem_type_parameters(4)
        # Set number of strain/stress components
        self._n_comp = len(self._comp_order_sym)
    # -------------------------------------------------------------------------
    def set_device(self, device_type):
        """Set device on which torch.Tensor is allocated.
        
        Parameters
        ----------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        if device_type in ('cpu', 'cuda'):
            if device_type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('PyTorch with CUDA is not available. '
                                   'Please set the model device type as CPU '
                                   'as:\n\n' + 'model.set_device(\'cpu\').')
            self._device_type = device_type
            self._device = torch.device(device_type)
        else:
            raise RuntimeError('Invalid device type.')
    # -------------------------------------------------------------------------
    def get_device(self):
        """Get device on which torch.Tensor is allocated.
        
        Returns
        -------
        device_type : {'cpu', 'cuda'}
            Type of device on which torch.Tensor is allocated.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        return self.device_type, self.device
    # -------------------------------------------------------------------------
    def forward(self, features_in):
        """Forward propagation.
        
        The tensor of input features must include the concatenated volumetric
        (features index 0) and the deviatoric components
        (features indices 1:1+n_comp) of the three-dimensional second-order
        strain/stress tensor.
        
        The tensor of output features is the strain/stress.
        
        Parameters
        ----------
        features_in : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_in) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_in) for batched input.

        Returns
        -------
        features_out : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, n_features_out) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, n_features_out) for batched input.
        """
        # Check input state features
        if not isinstance(features_in, torch.Tensor):
            raise RuntimeError('Input features were not provided as '
                               'torch.Tensor.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if batched input
        is_batched = features_in.dim() == 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batching dimension
        batch_dim = 1
        # Set batched dimension
        if not is_batched:
            features_in = features_in.unsqueeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check input state features
        if features_in.shape[-1] != 1 + self._n_comp:
            raise RuntimeError(f'Number of input features '
                               f'({features_in.shape[-1]}) must be equal to '
                               f'to one plus the number of strain/stress '
                               f'components ({1 + self._n_comp}), '
                               f'i.e., the volumetric and deviatoric '
                               f'components.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get input shape
        sequence_length, batch_size, n_features_in = features_in.shape
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate features sequences
        # (shape: sequence_length*batch_size x n_features_in)
        features_in = features_in.view(-1, n_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize output features (input deviatoric components)
        features_out = features_in[:, 1:]
        # Get volumetric component
        vol_comp = features_in[:, 0]
        # Loop over components
        for i, comp in enumerate(self._comp_order_sym):
            # Add volumetric component
            if comp[0] == comp[1]:
                features_out[:, i] += vol_comp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of output features
        n_features_out = features_out.shape[-1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Revert features sequences concatenation
        # (shape: sequence_length x batch_size x n_features_out)
        features_out = features_out.view(
            sequence_length, batch_size, n_features_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            features_out = features_out.squeeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return features_out