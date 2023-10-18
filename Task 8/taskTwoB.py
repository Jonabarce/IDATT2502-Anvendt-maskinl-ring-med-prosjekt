import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Environment settings
env = gym.make("CartPole-v1", render_mode="human")

# Hyperparameters
episodes = 1000
learning_rate = 0.001
gamma = 0.995
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_buffer_capacity = 10000

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*samples)

        # Convert observations to NumPy arrays and concatenate
        state = np.vstack([extract_array_from_observation(s) for s in state])
        action = np.vstack(action)
        reward = np.vstack(reward)
        next_state = np.vstack([extract_array_from_observation(s) for s in next_state])
        done = np.vstack(done)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Function to extract arrays from observations
def extract_array_from_observation(observation):
    if isinstance(observation, tuple) and len(observation) > 0:
        return observation[0]
    return observation

# Initialize DQN and Replay Buffer
policy_net = DQN(env.observation_space.shape[0], env.action_space.n).float()
target_net = DQN(env.observation_space.shape[0], env.action_space.n).float()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(replay_buffer_capacity)
criterion = nn.MSELoss()

# Select action with epsilon-greedy policy
def select_action(state, epsilon):
    # Check if state is a tuple, if so, retrieve the array from it
    if isinstance(state, tuple) or isinstance(state, list):
        state = state[0]

    state = torch.FloatTensor(state)

    # Check if you should explore
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Run through the network
        action_from_network = policy_net(state_tensor).argmax().item()
        return action_from_network

# Update DQN
def update_dqn(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward).unsqueeze(1)
    next_state = torch.FloatTensor(next_state)
    done = torch.FloatTensor(done).unsqueeze(1)

    q_values = policy_net(state).gather(1, action)
    next_q_values = target_net(next_state).max(1)[0].unsqueeze(1)
    expected_q_values = reward + (1 - done) * gamma * next_q_values

    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_cartpole():
    epsilon = epsilon_start
    total_steps = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0

        while True:
            action = select_action(state, epsilon)
            observation = env.step(action)
            next_state, reward, done, _, _ = observation
            replay_buffer.push(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            total_steps += 1

            if len(replay_buffer) > batch_size:
                update_dqn(batch_size)

            if total_steps % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Decay epsilon
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")

        if episode_reward >= 195:
            print(f"Solved in {episode} episodes!")
            break

    env.close()

if __name__ == "__main__":
    train_cartpole()
