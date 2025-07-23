import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Model(nn.Module):
    def __init__(self, trajectory_length=50):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

        self.states = []
        self.actions = []
        self.rewards = []
        self.trajectory_length = trajectory_length
        self.current_trajectory = 0

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state, reward):
        # Append reward from previous state-action pair
        self.rewards.append(reward)

        # Convert state to tensor and predict action
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_prediction = self.forward(state_tensor)

        # Detach action to store
        actual_action = action_prediction.detach().numpy()
        self.states.append(state)
        self.actions.append(actual_action)

        # Train when we have enough transitions
        if len(self.rewards) >= self.trajectory_length:
            self.train_with_reward_scaling()

        return action_prediction

    def train_with_reward_scaling(self):
        # Remove the first reward (it's meaningless — no action caused it)
        rewards = torch.tensor(self.rewards[:-1], dtype=torch.float32) ## Reward function takes care of the first reward
        states = torch.tensor(self.states[:-1], dtype=torch.float32)
        actions = torch.tensor(self.actions[:-1], dtype=torch.float32)

        # Normalize rewards
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 0:
            normalized_rewards = (rewards - reward_mean) / reward_std
        else:
            normalized_rewards = rewards - reward_mean

        # Forward pass to get predicted actions
        predictions = self.forward(states)

        # Compute per-sample MSE loss and scale by reward
        per_sample_loss = self.criterion(predictions, actions).mean(dim=1)
        weighted_loss = (per_sample_loss * normalized_rewards).mean()

        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Reset buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_trajectory += 1
