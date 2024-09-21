import matplotlib.pyplot as plt
import numpy as np

def extract_rewards(file_path):
    episodes = []
    rewards = []
    with open(file_path) as f:
        for line in f:
            try:
                parts = line.split('-')
                episode = int(parts[0].split()[1])
                reward = float(parts[1].split()[1])
                episodes.append(episode)
                rewards.append(reward)
            except (ValueError, IndexError):
                continue
    return episodes, rewards


# Extract times from the files
ddqn_reward_episodes, ddqn_rewards = extract_rewards('ddqn_reward.txt')
dqn_reward_episodes, dqn_rewards = extract_rewards('dqn_reward.txt')

# Plotting the rewards
plt.figure(figsize=(12, 6))
plt.plot(ddqn_reward_episodes, ddqn_rewards, label='DDQN', marker='o')
plt.plot(dqn_reward_episodes, dqn_rewards, label='DQN', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards by DDQN and DQN')
plt.legend()
plt.grid(True)
#plt.savefig('rewards_plot.png')
plt.show()

