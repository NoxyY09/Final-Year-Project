import os
import numpy as np
import matplotlib.pyplot as plt
from sumolib import checkBinary  # Finds the SUMO binary
import traci  # Interface for controlling SUMO via Python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# === Configuration ===
SUMO_CMD = [checkBinary("sumo"), "-c", "Brighton1.sumocfg"]  # Load the SUMO configuration file
TRAFFIC_LIGHT_ID = "14603342"  # ID of the traffic light to control
MAX_STEPS = 1000  # Max steps per simulation episode
N_EPISODES = 50  # Number of training episodes

# === Environment Class ===
class TrafficEnv:
    def __init__(self):
        self.action_space = 4  # Number of possible traffic light phases
        self.state_size = 2  # Features in state: queue length and waiting time
        self.traffic_data = []  # Used to store total rewards per episode

    def reset(self):
        """Reset the simulation and return the initial state."""
        if traci.isLoaded():
            traci.close()
        traci.start(SUMO_CMD)
        return self._get_state()

    def _get_state(self):
        """Extract the current state from the simulation."""
        controlled_lanes = traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID)
        monitored_edges = list(set(
            lane.split('_')[0] for lane in controlled_lanes if not lane.startswith(':')
        ))

        queue_length = sum(traci.edge.getLastStepVehicleNumber(e) for e in monitored_edges)
        waiting_time = sum(traci.edge.getWaitingTime(e) for e in monitored_edges)

        return np.array([queue_length, waiting_time], dtype=np.float32)

    def step(self, action):
        """Apply an action (change light phase), advance simulation, return result."""
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, action)
        traci.simulationStep()
        next_state = self._get_state()
        reward = -next_state[1]  # Reward is negative total waiting time (want to minimise it)
        done = traci.simulation.getTime() > MAX_STEPS
        return next_state, reward, done

    def close(self):
        """Stop the SUMO simulation."""
        traci.close()

# === Deep Q-Learning Agent ===
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()  # Neural network
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        """Build the neural network model."""
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

    def act(self, state):
        """Choose an action: explore randomly or exploit learned policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """Train the model using randomly sampled past experiences."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            target = reward + self.gamma * torch.max(self.model(next_state_tensor).detach()) * (1 - done)
            output = self.model(state_tensor)[action]
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# === Graphing Functions ===
def moving_average(data, window_size=5):
    """Smooth the reward curve using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_rewards(rewards):
    """Plot training progress over episodes."""
    smoothed = moving_average(rewards)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Waiting Time per Episode", color='lightblue')
    plt.plot(range(len(smoothed)), smoothed, label="Smoothed (Moving Average)", color='blue', linewidth=2)
    plt.title("Learning Curve: Reduction in Waiting Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Waiting Time (lower is better)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_baseline_vs_ai(baseline, post_ai):
    """Compare total waiting time per time step for baseline vs AI agent."""
    plt.figure(figsize=(10, 5))
    plt.plot(baseline, label="Without AI (Fixed Phase 0)", color='red', alpha=0.6)
    plt.plot(post_ai, label="With AI (Learned Policy)", color='green', alpha=0.8)
    plt.title("AI vs Baseline: Waiting Time per Simulation Step")
    plt.xlabel("Simulation Step")
    plt.ylabel("Waiting Time (vehicles)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main Program ===
if __name__ == "__main__":
    env = TrafficEnv()
    agent = DQNAgent(env.state_size, env.action_space)

    episode_rewards = []  # Track total reward (negative waiting time) for each episode

    # === Training ===
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.replay()
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{N_EPISODES} - Total Reward (Negative Wait): {total_reward:.2f}")

    # === Evaluation ===
    env.reset()
    baseline = [env.step(0)[1] for _ in range(MAX_STEPS)]  # No AI control (phase 0 every step)
    env.reset()
    post_ai = [env.step(agent.act(env._get_state()))[1] for _ in range(MAX_STEPS)]  # With AI control
    env.close()

    plot_training_rewards(episode_rewards)  # Plot learning curve
    plot_baseline_vs_ai(baseline, post_ai)  # Baseline graph to be called



