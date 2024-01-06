import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell

        Parameters: 
            input_dim: int, number of channels of input tensor
            hidden_dim: int, number of channels of hidden state
            kernel_size: (int, int), size of the convolutional kernel
            bias: bool, whether or not to add the bias
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Generate a multi-layer convolutional LSTM
    
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        torch.Tensor, (b,t,c,h,w) or (t,b,c,h,w)

    Output: (A tuple of two lists of length num_layers, or length 1 if return_all_layers is False)
        layer_output_list: list of lists of length T of each output
        last_state_list: list of last states, each element is tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim=32, hidden_dim=32, kernel_size=(3,3), num_layers=5,
                 batch_first=True, bias=True, return_all_layers=True, dropout=0):
        """
        Initialize ConvLSTM network
        
        Parameters:
            input_dim: int, number of channels of input tensor
            hidden_dim: int, number of channels of hidden state
            kernel_size: (int, int), size of the convolutional kernel
            num_layers: int, number of layers of ConvLSTM
            batch_first: bool, whether or not dimension 0 is the batch or not
            bias: bool, whether or not to add the bias
            return_all_layers: bool, whether or not to return all layers
            dropout: float, dropout rate    
        """
        super(ConvLSTM, self).__init__()
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            if layer_idx == self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list) or (isinstance(param, list) and len(param) == 1):
            param = [param] * num_layers
        return param


class ResConvLSTM(nn.Module):
    """
    Generate a multi-layer convolutional LSTM with residual connection

    Parameters:
    ----------
    n_dim: list of list of int
        each list contains dimensions of one block, of which the first element is input dimension and the rest are hidden dimensions for each layer
    kernel_size: list of list of int
        each list contains kernel sizes of one block, of which one element is kernel size for each layer
    dropout: float
        dropout rate

    Input:
    ----------
    input_data: torch.Tensor, (b,t,c,h,w)

    Output:
    ----------
    output_data: torch.Tensor, (b,t,c,h,w)
    """

    def __init__(self, n_dim=[[32,128,32]], kernel_size=[[5,3]], dropout=0.2):
        """
        Initialize ResConvLSTM network

        Parameters:
        ----------
        n_dim: list of list of int
            each list contains dimensions of one block, of which the first element is input dimension and the rest are hidden dimensions for each layer
        kernel_size: list of list of int
            each list contains kernel sizes of one block, of which one element is kernel size for each layer
        dropout: float
            dropout rate
        """
        super(ResConvLSTM, self).__init__()
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.n_layers = [len(k_list) for k_list in kernel_size]
        self.dropout = dropout

        cells = []
        convs = []
        for i in range(len(self.n_layers)):
            cells.append(ConvLSTM(n_dim[i][0], n_dim[i][1:], [(k,k) for k in kernel_size[i]], self.n_layers[i], dropout=dropout))
            if n_dim[i][-1] != n_dim[i][0]:
                convs.append(nn.Conv2d(in_channels=n_dim[i][0], out_channels=n_dim[i][-1], kernel_size=(1,1)))
            else:
                convs.append(nn.Identity())
        self.lstm = nn.Sequential(*cells)
        self.conv = nn.Sequential(*convs)
    
    def forward(self, input_data):
        for cell_num in range(len(self.lstm)):
            output, _ = self.lstm[cell_num](input_data)
            output = output[-1]
            input_data = torch.stack([self.conv[cell_num](input_data[:,t,:,:,:]) for t in range(input_data.shape[1])], dim=1)
            input_data = output + input_data
        return input_data