import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, use_cuda):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        use_cuda: bool
            Whether or not to put tensors on GPU
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.use_bias        = bias
        
        self.use_cuda = use_cuda
        # Don't use built-in bias of conv layers
        # self.conv.weights have shape (hidden_dim*4, input_dim+hidden_dim, kernel_w, kernel_h)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=False)
        # Set weights to something sane
        self.conv.weight.data.normal_(0, .01)
        
        # Use our own separate bias Variables
        # Order is (i, f, o, g)
        # Make sure everything is *i*nput to the state, nothing is *f*orgotten,
        # everything is *o*utput, and no bias for candidate C
        #TODO: These are put through a sigmoid so 1 does not have special value. Get more sane defaults?
        self.bias = Variable(torch.FloatTensor([1,1,1,0]), requires_grad=True)
        
        if self.use_cuda:
            # Module puts tensors on GPU in-place
            self.conv.cuda()
            # Variable.cuda() returns a copy of the variable-wrapped tensor on GPU
            self.bias = self.bias.cuda()

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
#         print('c_cur', c_cur)

        try:
            # TODO: Add optional peepholes like in original nowcasting paper
            combined = torch.cat((input_tensor, h_cur), dim=1)  # concatenate along channel axis
        # More useful notification for this common error
        except TypeError as e:
            raise Warning("TypeError when concatenating, you've probably given an incorrect tensor type. "
                          "Tried to concatenate input_tensor {} and h_cur {}"
                          .format(type(input_tensor.data), type(h_cur.data)))
        
        combined_conv = self.conv(combined)
        # TODO: Add peepholes, where i, f, o (elementwise) are dependent on c_cur
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i + self.bias[0])
#         print("i", i)
        f = torch.sigmoid(cc_f + self.bias[1])
#         print("f", f)
        o = torch.sigmoid(cc_o + self.bias[2])
#         print("cc_g", cc_g)
        g = torch.tanh(cc_g + self.bias[3])
#         print('g', g)
        
        c_next = f * c_cur + i * g
#         print('c_next', c_next)
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        hidden_states = (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                         Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        if self.use_cuda:
            hidden_states = [hs.cuda() for hs in hidden_states]
        return hidden_states


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, use_cuda=False):
        """
        Multi-layer unrolled Convolutional LSTM implementation.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int or (int, int) or ((int, int), ...) where len == num_layers
            Size of the convolutional kernel. Can be a single int for square kernel size equal for all layers,
            (int, int) for rectangular kernel equal for all layers, or a fully-specified list of tuples with first
            dimension equal to num_layers.
        bias: bool
            Whether or not to add the bias.
        use_cuda: bool
            Whether or not to put tensors on GPU (using nn.Module.cuda() does not work here, we are initializing
            hidden states during .forward(), which is nice because it gives us a flexible sequence length).
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer_kernel(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer_dim(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          use_cuda=use_cuda))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w) where t is sequence length
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        # hidden_state kwarg corresponds directly to last_state_list output
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)):
            raise ValueError('`kernel_size` must be tuple/list or list of lists')

    @staticmethod
    def _extend_for_multilayer_dim(param, num_layers):
        try:
            len(param)
            # If sequence, return it
            return list(param)
        except TypeError:
            # Extend it to a sequence with num_layers length
            return [param] * num_layers
    
    @staticmethod
    def _extend_for_multilayer_kernel(param, num_layers):
        try:
            # Verify if it's a sequence
            len(param)
            # It already is a list of tuples per layer
            if np.array(param).ndim == 2:
                return param
            # If not, we need to copy it num_layers times
            return [param] * num_layers
        except TypeError:
            # One value was given for a square kernel size equal for each layer
            return [[param]*2 for i in range(num_layers)]
