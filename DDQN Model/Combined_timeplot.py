import matplotlib.pyplot as plt
import numpy as np

def extract_times(file_path):
    episodes = []
    times = []
    with open(file_path) as f:
        for line in f:
            try:
                parts = line.split(':')
                episode = int(parts[0].split()[1])
                minutes = float(parts[1].strip().split()[0])
                episodes.append(episode)
                times.append(minutes)
            except (ValueError, IndexError):
                continue
    return episodes, times

# Extract times from the files
ddqn_episodes, ddqn_times = extract_times('ddqn_times.txt')
dqn_episodes, dqn_times = extract_times('dqn_times.txt')

# Plotting the cumulative times
plt.figure(figsize=(12, 6))
plt.plot(ddqn_episodes, ddqn_times, label='DDQN', marker='o')
plt.plot(dqn_episodes, dqn_times, label='DQN', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Total Time Taken (minutes)')
plt.title('Cumulative Total Time Taken by DDQN and DQN')
plt.legend()
plt.grid(True)
plt.show()
