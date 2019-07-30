import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,
                 hidden_sizes=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (int, array-like): number of neurons
                in each hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # format the hidden_sizes parameter
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes] * 2
        elif isinstance(hidden_sizes, (list, np.ndarray)):
            if len(hidden_sizes) == 1:
                self.hidden_sizes = [hidden_sizes[0]] * 2
            else:
                self.hidden_sizes = np.atleast_1d(hidden_sizes)
        else:
            raise ValueError('hidden_sizes should be of type int'
                             ' or array-like.')

        self.input_layer = nn.Linear(state_size, int(self.hidden_sizes[0]))
        self.middle_layers = nn.ModuleList(
            [nn.Linear(int(self.hidden_sizes[i - 1]),int(self.hidden_sizes[i]))
             for i in range(1, len(self.hidden_sizes))])
        self.output_layer = nn.Linear(int(hidden_sizes[-1]), action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.middle_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
