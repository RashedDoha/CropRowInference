import torch
import torch.nn as nn
from models.cnn import ConvNet
from models.rnn import GRUNet

class CropNet(nn.Module):
    """
        Conv-GRU based network for regressing the coefficients for three 2nd degree polynomials
        fitted through the central 3 crop rows under consideration
    """

    def __init__(self, batch_size, conv_out_dim, input_dim, hidden_dim, layer_dim, output_dim):
        super(CropNet, self).__init__()
        self.conv_out_dim = conv_out_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.cnn = ConvNet(conv_out_dim)
        self.rnn = GRUNet(batch_size, input_dim, hidden_dim, layer_dim, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], self.conv_out_dim//self.input_dim, -1).requires_grad_()
        x = self.rnn(x)
        return x