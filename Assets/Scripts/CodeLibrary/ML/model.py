import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import json
import os

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, trajectory_length=128):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.states = []
        self.actions = []
        self.rewards = []
        self.trajectory_length = trajectory_length
        self.current_trajectory = 0
        self.iteration = 0

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
        if random.random() < 0.1:
            action_idx = random.randint(0, 3)
        elif confidence < 0.4:
            # Use random action when confidence is low
            action_idx = random.randint(0, 3)
        else:
            # Use model's predicted action
            action_idx = torch.argmax(action_probs).item()
        
        # Store the state and actual action taken (one-hot encoded)
        actual_action = np.zeros(4)
        actual_action[action_idx] = 1.0
        self.states.append(state)
        self.actions.append(actual_action)
        
        # Store the reward from the previous state-action pair
        # (This reward corresponds to the action taken in the previous timestep)
        self.rewards.append(reward)

        # Train when we have enough transitions
        if self.second_last:
            self.train_with_reward_scaling()
        elif reward == -100:
            self.second_last = True
        elif len(self.states) >= self.trajectory_length:
            self.train_with_reward_scaling()

        # Return the action that was actually taken
        actions = ["up", "down", "left", "right"]
        return actions[action_idx]

    def train_with_reward_scaling(self):
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

        self.save_average_trajectory_reward(rewards)

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

    def save_average_trajectory_reward(self, rewards):
        """Save the average trajectory reward to a JSON file"""
        data_dir = "reward_trends"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filename = f"{data_dir}/average_reward_{int(time.time())}.json"
        
        if self.iteration == 0:
            # Create new file and overwrite
            if rewards:
                average_reward = sum(rewards) / len(rewards)
                with open(filename, 'w') as f:
                    json.dump({"average_reward": average_reward, "rewards": rewards.tolist()}, f, indent=2)
                # Generate graph after creating new file
                self.graph_reward_trends()
        else:
            # Append to existing file
            existing_rewards = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    existing_rewards = data["rewards"]
            
            # Append new rewards
            all_rewards = existing_rewards + rewards.tolist()
            
            if all_rewards:
                average_reward = sum(all_rewards) / len(all_rewards)
                with open(filename, 'w') as f:
                    json.dump({"average_reward": average_reward, "rewards": all_rewards}, f, indent=2)
                # Generate updated graph after appending
                self.graph_reward_trends()

    
    def graph_reward_trends(self):
        """Graph the reward trends"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            data_dir = "reward_trends"
            filename = f"{data_dir}/average_reward_{int(time.time())}.json"
            
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    rewards = data["rewards"]
                
                if rewards:
                    # Create the plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(rewards, 'b-', linewidth=1, alpha=0.7)
                    plt.scatter(range(len(rewards)), rewards, c='red', s=20, alpha=0.6)
                    
                    # Add moving average line
                    if len(rewards) > 10:
                        window_size = min(10, len(rewards) // 4)
                        moving_avg = []
                        for i in range(len(rewards)):
                            start = max(0, i - window_size // 2)
                            end = min(len(rewards), i + window_size // 2 + 1)
                            moving_avg.append(sum(rewards[start:end]) / (end - start))
                        plt.plot(moving_avg, 'g-', linewidth=2, alpha=0.8, label=f'Moving Average (window={window_size})')
                        plt.legend()
                    
                    plt.title('Reward Trends Over Time')
                    plt.xlabel('Training Step')
                    plt.ylabel('Reward')
                    plt.grid(True, alpha=0.3)
                    
                    # Save the plot
                    plot_filename = f"{data_dir}/reward_trends_plot.png"
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Reward trends plot saved to: {plot_filename}")
                    
        except ImportError:
            print("matplotlib not available, skipping reward trend plotting")
        except Exception as e:
            print(f"Error creating reward trend plot: {e}")


    def save_training_data(self):
        """Save the current training data (rewards, states, actions) to a JSON file"""
        # Create data directory if it doesn't exist
        data_dir = "training_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Prepare data for saving
        training_data = {
            "trajectory_number": self.current_trajectory,
            "timestamp": time.time(),
            "rewards": [float(r) for r in self.rewards],
            "states": [state[1] for state in self.states],  # Only save y position (index 1)
            "actions": [action.tolist() if hasattr(action, 'tolist') else action for action in self.actions],
            "trajectory_length": len(self.states)
        }
        
        # Save to file with timestamp
        filename = f"{data_dir}/training_data_trajectory_{self.current_trajectory}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Training data saved to: {filename}")
