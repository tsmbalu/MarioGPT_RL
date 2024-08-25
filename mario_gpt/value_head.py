import torch.nn as nn


class ValueHead(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.liner = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.liner(hidden_states)
        return output.squeeze(-1)



