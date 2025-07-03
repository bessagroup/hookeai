"""Multi-layer gated recurrent unit (GRU) recurrent neural network model.

Custom implementation of GRUCell and GRU compatible with VMAPs.

Classes
-------
GRUCell(torch.nn.Module)
    Gated Recurrent Unit (GRU) cell.
GRU(torch.nn.Module)
    Multi-layer gated recurrent unit (GRU) recurrent neural network model.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import time
# Third-party
import torch
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class GRUCell(torch.nn.Module):
    """Gated Recurrent Unit (GRU) cell.
    
    Attributes
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of output features.
    bias : bool
        If True, then consider bias weights as learnable parameters.
    num_chunks : int
        Number of gates.
    w_ih : torch.Tensor(2d)
        Input-to-hidden weights stored as torch.Tensor(2d) of shape
        (3*hidden_size, input_size).
    w_hh : torch.Tensor(2d)
        Hidden-to-hidden weights stored as torch.Tensor(2d) of shape
        (3*hidden_size, input_size).
    b_ih : torch.Tensor(2d)
        Hidden-to-hidden bias weights stored as torch.Tensor(2d) of shape
        (3*hidden_size).
    b_hh : torch.Tensor(2d)
        Hidden-to-hidden bias weights stored as torch.Tensor(2d) of shape
        (3*hidden_size).
    device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    forward(self, input, hx=None)
        Forward propagation.
    """
    def __init__(self, input_size, hidden_size, bias=True, device=None):
        """Constructor.
        
        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of output features.
        bias : bool, default=True
            If True, then consider bias weights.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(GRUCell, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Set architecture parameters
        self.bias = bias
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        self.device = device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of chunks
        self.num_chunks = 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set weights (input-to-hidden)
        self.w_ih = torch.nn.Parameter(
            torch.empty((self.num_chunks*hidden_size, input_size),
                        device=self.device))
        # Set weights (hidden-to-hidden)
        self.w_hh = torch.nn.Parameter(
            torch.empty((self.num_chunks*hidden_size, hidden_size),
                        device=self.device))
        # Set biases
        if self.bias:
            # Set bias (input-to-hidden)
            self.b_ih = torch.nn.Parameter(
                torch.empty(self.num_chunks*hidden_size, device=self.device))
            # Set bias (hidden-to-hidden)
            self.b_hh = torch.nn.Parameter(
                torch.empty(self.num_chunks*hidden_size, device=self.device))
        else:
            # Set bias (input-to-hidden)
            self.b_ih = \
                torch.zeros(self.num_chunks*hidden_size, device=self.device)
            # Set bias (hidden-to-hidden)
            self.b_hh = \
                torch.zeros(self.num_chunks*hidden_size, device=self.device)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize parameters
        self.reset_parameters()
    # -------------------------------------------------------------------------
    def reset_parameters(self):
        """Initialize learnable parameters."""
        # Initialize uniform distribution bounds
        if self.hidden_size > 0:
            stdv = 1.0 / math.sqrt(self.hidden_size)
        else:
            stdv = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over parameters
        for parameter in self.parameters():
            # Initialize from uniform distribution
            torch.nn.init.uniform_(parameter, -stdv, stdv)
    # -------------------------------------------------------------------------
    def forward(self, input, hx=None):
        """Forward propagation.
        
        Parameters
        ----------
        input : torch.Tensor
            Tensor of input features stored as torch.Tensor(1d) of shape
            (input_size) for unbatched input or torch.Tensor(2d) of shape
            (batch_size, input_size) for batched input.
        hx : torch.Tensor, default=None
            Tensor of initial hidden state features stored as torch.Tensor(1d)
            of shape (hidden_size) for unbatched input or torch.Tensor(2d) of
            shape (batch_size, hidden_size) for batched input. If None, then
            defaults to zero.
        
        Returns
        -------
        h_updated : torch.Tensor
            Tensor of updated hidden state features stored as torch.Tensor(1d)
            of shape (hidden_size) for unbatched input or torch.Tensor(2d) of
            shape (batch_size, hidden_size) for batched input.
        """
        # Check if batched input
        is_batched = input.dim() == 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batched dimension
        if not is_batched:
            input = input.unsqueeze(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize hidden state features
        if hx is None:
            hx = torch.zeros((input.size(0), self.hidden_size),
                             device=input.device)
        else:
            # Set batched dimension
            if not is_batched:
                hx = hx.unsqueeze(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute input tensor linear transformations
        gi = torch.mm(input, self.w_ih.t()) + self.b_ih
        # Compute hidden tensor linear transformations
        gh = torch.mm(hx, self.w_hh.t()) + self.b_hh
        # Split input and hidden tensor linear transformations
        i_r, i_i, i_n = torch.chunk(gi, chunks=self.num_chunks, dim=1)
        h_r, h_i, h_n = torch.chunk(gh, chunks=self.num_chunks, dim=1)
        # Compute update gate output
        # (shape: batch_size x hidden_size)
        update_gate = torch.sigmoid(i_i + h_i)
        # Compute reset gate output
        # (shape: batch_size x hidden_size)
        reset_gate = torch.sigmoid(i_r + h_r)
        # Compute new gate output (new candidate hidden tensor)
        # (shape: batch_size x hidden_size)
        new_gate = torch.tanh(i_n + reset_gate*h_n)
        # Compute updated hidden tensor
        # (shape: batch_size x hidden_size)
        h_updated = update_gate*(hx - new_gate) + new_gate        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            h_updated = h_updated.squeeze(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return h_updated
# =============================================================================
class GRU(torch.nn.Module):
    """Multi-layer gated recurrent unit (GRU) recurrent neural network model.
    
    It is assumed that batched input and output tensors are provided with
    shape (sequential_length, batch_size, n_features), corresponding to the
    default setting batch_first=False.
    
    Dropout layers and bidirectionality are not implemented.
    
    Attributes
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of output features.
    bias : bool
        If True, then consider bias weights as learnable parameters.
    num_layers : int, default=1
        Number of recurrent layers. A number of recurrent layers greater than 1
        results in a stacked GRU (output of GRU in each time t is the input of
        next GRU).
    device : torch.device
        Device on which torch.Tensor is allocated.

    Methods
    -------
    forward(self, input, hx=None)
        Forward propagation.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 device=None):
        """Constructor.
        
        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_size : int
            Number of output features.
        num_layers : int, default=1
            Number of recurrent layers. A number of recurrent layers greater
            than 1 results in a stacked GRU (output of GRU in each time t is
            the input of next GRU).
        bias : bool, default=True
            If True, then consider bias weights.
        device : torch.device
            Device on which torch.Tensor is allocated.
        """
        # Initialize from base class
        super(GRU, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Set architecture parameters
        self.bias = bias
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of layers
        self.num_layers = num_layers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set device
        self.device = device
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize GRU layers
        self.gru_layers = torch.nn.ModuleList()
        # Set initial GRU layer (input to hidden)
        self.gru_layers.append(GRUCell(self.input_size, self.hidden_size,
                                       bias=self.bias, device=self.device))
        # Set remaining GRU layers (hidden to hidden)
        for _ in range(1, num_layers):
            self.gru_layers.append(GRUCell(self.hidden_size, self.hidden_size,
                                    bias=self.bias, device=self.device))
    # -------------------------------------------------------------------------
    def forward(self, input, hx=None):
        """Forward propagation.
        
        Parameters
        ----------
        input : torch.Tensor
            Tensor of input features stored as torch.Tensor(2d) of shape
            (sequence_length, input_size) for unbatched input or
            torch.Tensor(3d) of shape (sequence_length, batch_size, input_size)
            for batched input.
        hx : torch.Tensor, default=None
            Tensor of initial hidden state features stored as torch.Tensor(2d)
            of shape (num_layers, hidden_size) for unbatched input or
            torch.Tensor(3d) of shape (num_layers, batch_size, hidden_size) for
            batched input.
        
        Returns
        -------
        output : torch.Tensor
            Tensor of output features stored as torch.Tensor(2d) of shape
            (sequence_length, hidden_size) for unbatched input or
            torch.Tensor(3d) of shape
            (sequence_length, batch_size, hidden_size) for batched input.
            Corresponds to the tensor of hidden state features output from the
            last GRU layer.
        h_n : torch.Tensor
            Tensor of final multi-layer GRU hidden state features stored as
            torch.Tensor(2d) of shape (num_layers, hidden_size) for unbatched
            input or torch.Tensor(3d) of shape
            (num_layers, batch_size, hidden_size) for batched input.
        """
        # Get sequence length
        n_time = input.size(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batching dimension
        batch_dim = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if batched input
        is_batched = input.dim() == 3
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set batched dimension
        if not is_batched:
            input = input.unsqueeze(batch_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get batch size
        batch_size = input.size(batch_dim)
        # Initialize hidden state features
        if hx is None:
            hx = torch.zeros((self.num_layers, batch_size, self.hidden_size),
                             device=input.device)
        else:
            # Set batched dimension
            if not is_batched:
                hx = hx.unsqueeze(1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize GRU layers (current) hidden state
        h_layers = [hx[l, :, :] for l in range(self.num_layers)]
        # Initialize output features
        output_times = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over time steps
        for t in range(n_time):
            # Loop over GRU layers
            for l in range(self.num_layers):
                # Get GRU layer input for current time step
                if l == 0:
                    layer_input = input[t, :, :]
                else:
                    layer_input = h_layers[l - 1]
                # Compute GRU layer updated hidden state for current time step
                h_updated = self.gru_layers[l](layer_input, h_layers[l])
                # Update GRU layer (current) hidden state
                h_layers[l] = h_updated
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store last GRU layer output for current time step
            output_times.append(h_updated)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build tensor of output features
        output = torch.stack(output_times, dim=0)
        # Build tensor of final multi-layer GRU hidden state features
        h_n = torch.stack(h_layers, dim=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove batched dimension
        if not is_batched:
            output = output.squeeze(1)
            h_n = h_n.squeeze(1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return output, h_n
# =============================================================================
if __name__ == '__main__':
    # Set function timer
    def function_timer(function, args, n_calls=1):
        # Initialize total execution time
        total_time = 0
        # Loop over number of function calls
        for i in range(n_calls):
            # Set initial call time
            t0 = time.time()
            # Call function
            function(*args)
            # Add to total execution time
            total_time += time.time() - t0
        # Compute average time per function call
        avg_time_call = total_time/n_calls
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return avg_time_call
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set backward timer
    def backward_timer(scalar):
        # Initialize total execution time
        t0 = time.time()
        # Backward propagation
        scalar.backward()
        # Set total execution time
        total_time = time.time() - t0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return total_time
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set seed for reproducibility
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random number generators initialization for reproducibility
    seed = 0
    set_seed(seed)
    # Set device
    device = torch.device('cuda')
    # Set batch testing
    is_batched = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set parameter option
    option = ('simple', 'complex')[0]
    # Set parameters
    if option == 'complex':
        # Set number of features
        input_size = 6
        hidden_size = 500
        # Set sequence length
        sequence_length = 100
        # Set batch size (number of time series)
        batch_size = 1
        # Set number of GRU layers
        num_layers = 3
    else:
        # Set number of features
        input_size = 6
        hidden_size = 5
        # Set sequence length
        sequence_length = 10
        # Set batch size (number of time series)
        batch_size = 4
        # Set number of GRU layers
        num_layers = 3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set random input time series data
    input = torch.randn((sequence_length, batch_size, input_size),
                        device=device)
    # Set random initial hidden state
    hx = torch.randn((num_layers, batch_size, hidden_size),
                     device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display
    print(f'\nTesting: Torch GRU Cell vs Custom GRU Cell' + '\n'
          + '-'*len('\nTesting: Torch GRU Cell vs Custom GRU Cell'))
    print(f'\n  > Device: {device}')
    print(f'\n  > Batched input/output: {is_batched}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize PyTorch GRU Cell
    set_seed(seed)
    gru_cell_torch = torch.nn.GRUCell(input_size, hidden_size, bias=True,
                                      device=device)    
    # Initialize Custom GRU Cell
    set_seed(seed)
    gru_cell_custom = GRUCell(input_size, hidden_size, bias=True,
                              device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set batched/unbatched data
    if is_batched:
        input_cell = input[0, :, :].clone()
        hx_cell = hx[0, :, :].clone()
    else:
        input_cell = input[0, 0, :].clone()
        hx_cell = hx[0, 0, :].clone()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (updated hidden state)
    h_updated_torch = gru_cell_torch(input_cell, hx_cell)
    # Forward propagation (updated hidden state)
    h_updated_custom = gru_cell_custom(input_cell, hx_cell)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare results
    if not torch.allclose(h_updated_torch, h_updated_custom):
        print(f'\n  > Matching results? FALSE')
        raise RuntimeError('Torch and Custom GRU Cells results do not match!')
    else:
        print(f'\n  > Matching results? TRUE')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check backward propagation time
    backward_time_torch = backward_timer(torch.sum(h_updated_torch))
    backward_time_custom = backward_timer(torch.sum(h_updated_custom))
    backward_time_ratio = backward_time_custom/backward_time_torch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check average time per call
    avg_time_call_torch = function_timer(gru_cell_torch,
                                         (input_cell, hx_cell),
                                         n_calls=1000)
    avg_time_call_custom = function_timer(gru_cell_custom,
                                          (input_cell, hx_cell),
                                          n_calls=1000)
    avg_time_call_ratio = avg_time_call_custom/avg_time_call_torch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare computational times
    print(f'\n  > Forward propagation times:')
    print(f'\n    - Avg. time per call (Torch):  {avg_time_call_torch:.4e}')
    print(f'\n    - Avg. time per call (Custom): {avg_time_call_custom:.4e}')
    print(f'\n    - Custom/Torch = {avg_time_call_ratio:.2f} ')
    print(f'\n\n  > Backward propagation times:')
    print(f'\n    - Time (Torch): {backward_time_torch:.4e}')
    print(f'\n    - Time (Custom): {backward_time_custom:.4e}')
    print(f'\n    - Custom/Torch = {backward_time_ratio:.2f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display
    print(f'\n\nTesting: Torch GRU vs Custom GRU' + '\n'
          + '-'*len('\nTesting: Torch GRU vs Custom GRU'))
    print(f'\n  > Device: {device}')
    print(f'\n  > Batched input/output: {is_batched}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize PyTorch GRU
    set_seed(seed)
    gru_torch = torch.nn.GRU(input_size, hidden_size, num_layers=num_layers,
                             bias=True, batch_first=False, device=device)
    # Initialize Custom GRU
    set_seed(seed)
    gru_custom = GRU(input_size, hidden_size, num_layers=num_layers,
                     bias=True, device=device)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set batched/unbatched data
    if is_batched:
        pass
    else:
        input = input[:, 0, :].clone()
        hx = hx[:, 0, :].clone()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    output_torch, h_n_torch = gru_torch(input, hx)
    # Forward propagation
    output_custom, h_n_custom = gru_custom(input, hx)    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Notes:
    # The absolute difference in the following comparison seems to increase
    # as the number of sequential computation increases (increasing number of
    # layers or sequence length). Suspect that this may result from the
    # accumulation of error because (1) matching results are obtained for
    # GRU Cell irrespective of parameters and (ii) matching results are
    # obtained for GRU with parameters leading to lower number of sequential
    # operations (layers or time).
    #
    # Example of GRU parameters leading to matching results:
    #   input_size = 6
    #   hidden_size = 5
    #   sequence_length = 10
    #   batch_size = 4
    #   num_layers = 3
    #
    # Set raise error when mismatching results
    is_raise_error = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare results
    if not torch.allclose(output_torch, output_custom, atol=1e-08):
        print(f'\n  > Output matching results? FALSE')
        if is_raise_error:
            raise RuntimeError('Torch and Custom GRUs output results do not '
                               'match!')
    elif not torch.allclose(h_n_torch, h_n_custom, atol=1e-08):
        print(f'\n  > Layers hidden states matching results? FALSE')
        if is_raise_error:
            raise RuntimeError('Torch and Custom GRUs layers hidden states '
                               'results do not match!')
    else:
        print(f'\n  > Matching results? TRUE')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check backward propagation time
    backward_time_torch = backward_timer(torch.sum(output_torch))
    backward_time_custom = backward_timer(torch.sum(output_custom))
    backward_time_ratio = backward_time_custom/backward_time_torch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check average time per call
    avg_time_call_torch = function_timer(gru_torch, (input, hx),
                                         n_calls=100)
    avg_time_call_custom = function_timer(gru_custom, (input, hx),
                                          n_calls=100)
    avg_time_call_ratio = avg_time_call_custom/avg_time_call_torch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compare computational times
    print(f'\n  > Forward propagation times:')
    print(f'\n    - Avg. time per call (Torch):  {avg_time_call_torch:.4e}')
    print(f'\n    - Avg. time per call (Custom): {avg_time_call_custom:.4e}')
    print(f'\n    - Custom/Torch = {avg_time_call_ratio:.2f}')
    print(f'\n\n  > Backward propagation times:')
    print(f'\n    - Time (Torch): {backward_time_torch:.4e}')
    print(f'\n    - Time (Custom): {backward_time_custom:.4e}')
    print(f'\n    - Custom/Torch = {backward_time_ratio:.2f}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test vectorizing maps (only for batched input/output)
    if is_batched:
        # Display
        print(f'\n\nTesting: Custom GRU Cell vs Custom GRU Cell VMAP' + '\n'
            + '-'*len('\nTesting: Custom GRU Cell vs Custom GRU Cell VMAP'))
        print(f'\n  > Device: {device}')
        print(f'\n  > Batched input/output: {is_batched}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized GRU cell (batch along time)
        vmap_gru_cell_custom = \
            torch.vmap(gru_cell_custom, in_dims=(0, 0), out_dims=(0,))
        # Forward propagation (updated hidden state)
        h_updated_custom_vmap = vmap_gru_cell_custom(input_cell, hx_cell)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compare results
        if not torch.allclose(h_updated_custom, h_updated_custom_vmap):
            print(f'\n  > Matching results? FALSE')
            raise RuntimeError('Custom and Custom VMAP GRU Cells results do '
                               'not match!')
        else:
            print(f'\n  > Matching results? TRUE')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Backward propagation
        torch.sum(h_updated_custom_vmap).backward()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display
        print(f'\n\nTesting: Custom GRU vs Custom GRU VMAP' + '\n'
            + '-'*len('\nTesting: Custom GRU vs Custom GRU VMAP'))
        print(f'\n  > Device: {device}')
        print(f'\n  > Batched input/output: {is_batched}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set vectorized GRU (batch along time series)
        vmap_gru_custom = \
            torch.vmap(gru_custom, in_dims=(1, 1), out_dims=(1, 1))
        # Forward propagation (updated hidden state)
        output_custom_vmap, h_n_custom_vmap = vmap_gru_custom(input, hx)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compare results
        if not torch.allclose(output_custom, output_custom_vmap):
            print(f'\n  > Output matching results? FALSE')
            if is_raise_error:
                raise RuntimeError('Torch and Custom GRUs output results do '
                                   'not match!')
        elif not torch.allclose(h_n_custom, h_n_custom_vmap):
            print(f'\n  > Layers hidden states matching results? FALSE')
            if is_raise_error:
                raise RuntimeError('Torch and Custom GRUs layers hidden '
                                   'states results do not match!')
        else:
            print(f'\n  > Matching results? TRUE')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print()