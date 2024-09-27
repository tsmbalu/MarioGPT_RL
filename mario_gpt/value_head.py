"""
Author: Balasubramani Murugan

This script defines a ValueHead neural network module used to predicts a single scalar
value given the hidden states as input.
"""
import torch.nn as nn


class ValueHead(nn.Module):
    """
       A simple value head module that outputs a scalar value from the hidden states of a model.

        hidden_size (int): The size of the hidden layer from the preceding model layers.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.liner = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        """
        Forward pass for the ValueHead. It takes the hidden states and outputs a single scalar value.

        @param hidden_states: hidden states from the preceding model layers
        @return: scalar value
        """
        output = self.liner(hidden_states)
        return output.squeeze(-1)



