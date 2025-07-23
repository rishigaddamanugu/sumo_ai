import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, trajectory_length=50):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

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
        
        # Apply softmax to get probabilities
        x = torch.softmax(x, dim=0)
        return x

    def predict(self, state, reward):
        # Append reward from previous state-action pair
        self.rewards.append(reward)

        # Convert state to tensor and predict action
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.forward(state_tensor)
        
        # Calculate confidence (max probability)
        confidence = torch.max(action_probs).item()
        
        # Use random actions if confidence is low (less than 0.4)
        if confidence < 0.4:
            actions = ["up", "down", "left", "right"]
            return random.choice(actions)
        
        # Get the action that was actually taken
        if confidence < 0.4:
            # Random action was taken
            action_idx = random.randint(0, 3)
        else:
            # Model action was taken
            action_idx = torch.argmax(action_probs).item()
        
        # Store the actual action taken (one-hot encoded)
        actual_action = np.zeros(4)
        actual_action[action_idx] = 1.0
        self.states.append(state)
        self.actions.append(actual_action)

        # Train when we have enough transitions
        if len(self.rewards) >= self.trajectory_length:
            self.train_with_reward_scaling()

        # Convert to action string
        action_idx = torch.argmax(action_probs).item()
        actions = ["up", "down", "left", "right"]
        return actions[action_idx]

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
