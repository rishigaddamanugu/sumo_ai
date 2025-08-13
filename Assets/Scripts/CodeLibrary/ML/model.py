import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import json
import os


###################################
# MAJOR FLAW in TWO AGENT SYSTEM
# usually, because states and rewards are misaligned, we offset by 1
# Since both agents query the same model, offsetting by 1 doesn't work because the alignment depends on how many agents there are
# We need some way to know how many agents there are and mod to offset before training
# Alternatively, we can just use one model per agent, but that means we need to launch a new model for each agent
###################################

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, trajectory_length=16):
        super(Model, self).__init__()

        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.states = []
        self.actions = []
        self.rewards = []
        self.trajectory_length = trajectory_length
        self.current_trajectory = 0
        self.iteration = 0
        self.total_rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss(reduction='none')
        self.second_last = False

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply softmax to get probabilities
        x = torch.softmax(x, dim=0)
        return x

    def predict(self, state, reward):
        # Convert state to tensor and predict action
        self.iteration += 1
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.forward(state_tensor)
        
        # Calculate confidence (max probability)
        confidence = torch.max(action_probs).item()
        
        # Determine which action to take
        # 10% chance to take random action for exploration
        if confidence < 0.5:
            # Use random action when confidence is low
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Use model's predicted action
            action_idx = torch.argmax(action_probs).item()
        
        # Store the state and actual action taken (one-hot encoded)
        actual_action = np.zeros(self.action_dim)
        actual_action[action_idx] = 1.0
        self.states.append(state)
        self.actions.append(actual_action)
        
        # Store the reward from the previous state-action pair
        # (This reward corresponds to the action taken in the previous timestep)
        self.rewards.append(reward)

        # Train when we have enough transitions
        if len(self.states) >= self.trajectory_length:
            self.train_with_reward_scaling()

        # Return the action that was actually taken
        actions = ["forward", "backward", "left", "right", "jump"]
        return actions[action_idx]

    def train_with_reward_scaling(self, epochs=10):
        print("TRAINING MODEL!!!!!!")
        time.sleep(10)

        # Safety check: ensure we have enough data to train
        if len(self.states) < 2 or len(self.actions) < 2 or len(self.rewards) < 2:
            return
        
        # # Save the training data
        # self.save_training_data()
        
        # # Add 40-second delay
        # print("Training completed. Waiting 40 seconds before next training cycle...")
        # time.sleep(40)
        
        # Convert lists to numpy arrays first for better performance
        # Note: rewards[1:] corresponds to states[:-1] and actions[:-1]
        # because reward is received after taking an action
        rewards = torch.tensor(np.array(self.rewards[1:]), dtype=torch.float32)
        states = torch.tensor(np.array(self.states[:-1]), dtype=torch.float32)  # Remove last state
        actions = torch.tensor(np.array(self.actions[:-1]), dtype=torch.float32)  # Remove last action


        print("Rewards: ", rewards)
        print("States: ", states)
        print("Actions: ", actions)

        # Normalize rewards
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 0:
            normalized_rewards = (rewards - reward_mean) / reward_std
        else:
            normalized_rewards = rewards - reward_mean

        # Training loop with epochs
        for epoch in range(epochs):
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

        self.total_rewards.append(rewards)