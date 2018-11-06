import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    This module defines an policy model as a neural network
    (i.e. takes a state as an input and returns (unnormalized) probability
    to take possible actions).

    """
    def __init__(self, state_size, action_size, seed = 123):
        """
        For initialization.

        (input)
        - state_size (int): size of each state
        - action_size (int): dimension of the action space
        - seed (int): random seed

        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed) # random seed
        self.state_size = state_size # size of a state
        self.action_size = action_size  # dim of action space
        self.hidden_sizes = [128, 128] # the numbers of units in the hidden layers

        # list of input dim, numbers of units in hidden layers and ouput dim
        list_dims= [self.state_size] + self.hidden_sizes + [self.action_size]
        list_dims_pair = zip(list_dims[0:-1], list_dims[1:])

        # list of fully connected layers
        self.layers = nn.ModuleList([])
        for in_dim, out_dim in list_dims_pair:
            self.layers.extend([nn.Linear(in_dim, out_dim)])

    def forward(self, state):
        """
        This method defines the forward pass for the this neural network model.

        (input)
        - state (tensor with dim = state_size): a state vector

        (output)
        - tensor with dim = action_size

        """
        x = state
        for i in range(len(self.layers)-1):
            x = self.layers[i](x) # fully connected layer
            x = F.relu(x) # Relu activation
        x = self.layers[-1](x)

        return x
