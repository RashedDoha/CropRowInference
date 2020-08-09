import torch
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUNet, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # GRU layers
        self.gru_1l = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True) # batch_first=True (batch_dim, seq_dim, feature_dim)
        self.gru_1r = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.gru_2l = nn.GRU(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.gru_2r = nn.GRU(hidden_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.h0 = nn.Parameter(torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_())

    def forward(self, x):
        # Initialize hidden state with zeros

        o_l, h_l = self.gru_1l(x, self.h0.detach())
        o_r, h_r = self.gru_1r(x, self.h0.detach())
        o, h = torch.stack([o_l, o_r], dim=2).sum(dim=2), torch.stack([h_l, h_r], dim=2).sum(dim=2)
        o_l, h_l = self.gru_2l(o, h.detach())
        o_r, h_r = self.gru_2r(o, h.detach())
        out,hidden = torch.cat([o_l, o_r], dim=2), torch.stack([h_l, h_r], dim=2).sum(dim=2)

        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out