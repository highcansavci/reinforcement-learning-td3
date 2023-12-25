import torch
import os
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, name, chkp_dir="models/td3"):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkp_dir, name + "_td3")

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_action_value = self.fc1(torch.cat([state, action], dim=1))
        state_action_value = F.relu(state_action_value)
        state_action_value = self.fc2(state_action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

