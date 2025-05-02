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
import time

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
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_training_rewards(rewards):
    """Plot training progress over episodes."""
    smoothed = moving_average(rewards)
    plt.figure(1)  # Create first figure
    plt.plot(rewards, label="Total Waiting Time per Episode", color='lightblue')
    plt.plot(range(len(smoothed)), smoothed, label="Smoothed (Moving Average)", color='blue', linewidth=2)
    plt.title("Learning Curve: Reduction in Waiting Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Waiting Time (lower is better)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_baseline_vs_ai(baseline, post_ai):
    """Compare total waiting time per time step for baseline vs AI agent."""
    plt.figure(2)  # Create second figure
    plt.plot(baseline, label="Without AI (Fixed Phase 0)", color='red', alpha=0.6)
    plt.plot(post_ai, label="With AI (Learned Policy)", color='green', alpha=0.8)
    plt.title("AI vs Baseline: Waiting Time per Simulation Step")
    plt.xlabel("Simulation Step")
    plt.ylabel("Waiting Time (vehicles)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# === Testing Functions ===
def test_speed(agent, env, max_steps=MAX_STEPS):
    """Test the speed of the AI agent's decision-making."""
    start_time = time.time()
    state = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    return execution_time


def test_quality(agent, env, max_steps=MAX_STEPS):
    """Compare AI vs Baseline performance."""
    print("Running AI vs Baseline test")

    # Test baseline (fixed phase 0)
    env.reset()
    baseline_rewards = []
    for _ in range(max_steps):
        _, reward, done = env.step(0)  # Always use phase 0
        baseline_rewards.append(reward)
        if done:
            break

    # Test AI control
    env.reset()
    ai_rewards = []
    state = env._get_state()
    for _ in range(max_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        ai_rewards.append(reward)
        state = next_state
        if done:
            break

    # Plot comparison
    plot_baseline_vs_ai(baseline_rewards, ai_rewards)
    return baseline_rewards, ai_rewards


def test_user_acceptance(agent, env):
    """Run user acceptance tests."""
    print("Running User Acceptance Testing (UAT) for AI system")

    # Test environment reset
    state = env.reset()
    assert state is not None, "Environment reset failed"
    print("✓ Environment reset test passed")

    # Test agent action
    action = agent.act(state)
    assert action is not None and 0 <= action < env.action_space, "Agent action test failed"
    print("✓ Agent action test passed")

    # Test simulation step
    next_state, reward, done = env.step(action)
    assert next_state is not None, "Simulation step test failed"
    print("✓ Simulation step test passed")

    print("All UAT tests passed successfully!")


# === Main Program ===
if __name__ == "__main__":
    env = TrafficEnv()
    agent = DQNAgent(env.state_size, env.action_space)

    # === Training ===
    print("Starting training...")
    episode_rewards = []  # Track total reward (negative waiting time) for each episode
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
    print("Training completed!")

    # === Testing ===
    print("\nStarting testing phase...")

    # Run speed test
    print("\nRunning speed test...")
    speed = test_speed(agent, env)
    print(f"Speed test completed in {speed:.4f} seconds")

    # Run quality test
    print("\nRunning quality test...")
    baseline_rewards, ai_rewards = test_quality(agent, env)
    print("Quality test completed")

    # Run user acceptance tests
    print("\nRunning user acceptance tests...")
    test_user_acceptance(agent, env)

    # === Graphing ===
    print("\nGenerating plots...")
    plot_training_rewards(episode_rewards)  # Plot learning curve
    plot_baseline_vs_ai(baseline_rewards, ai_rewards)  # Baseline vs AI comparison
    plt.show()  # Show both plots at once
    print("Plots generated successfully!")

    # Clean up
    env.close()
    print("\nAll tests completed successfully!")
