import matplotlib.pyplot as plt
import numpy as np

def extract_losses(file_path):
    episodes = []
    losses = []
    with open(file_path) as f:
        for line in f:
            try:
                parts = line.split('-')
                episode = int(parts[0].split()[1])
                loss = float(parts[1].split()[2])
                episodes.append(episode)
                losses.append(loss)
            except (ValueError, IndexError):
                continue
    return episodes, losses

# Extract data from the files
ddqn_loss_episodes, ddqn_losses = extract_losses('ddqn_loss.txt')
dqn_loss_episodes, dqn_losses = extract_losses('dqn_loss.txt')

# Plotting the losses
plt.figure(figsize=(12, 6))
plt.plot(ddqn_loss_episodes, ddqn_losses, label='DDQN', marker='o')
plt.plot(dqn_loss_episodes, dqn_losses, label='DQN', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Losses')
plt.title('Losses by DDQN and DQN')
plt.legend()
plt.grid(True)
#plt.savefig('losses_plot.png')
plt.show()