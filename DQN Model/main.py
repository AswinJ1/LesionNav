import numpy as np
import torch
from environment import BrainEnv
from agent import DQNAgent
import pygame
import os
import matplotlib.pyplot as plt
import time
from torchsummary import summary

def plot_real_time(rewards, losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    
    plt.subplot(122)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Time')
    
    plt.pause(0.001)
    plt.show()

def plot_cumulative_time(epochs, cumulative_times):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, cumulative_times, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Time (in minutes)')
    plt.title('Cumulative Time to Complete Milestone Episodes')
    plt.grid(True)
    plt.show()

def format_duration(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes} min {seconds} sec"

def train(env, agent, num_episodes=1050, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    epsilon = epsilon_start
    cumulative_times = []
    total_duration = 0  # Initialize total duration
    milestones = list(range(50, num_episodes + 1, 50))  # Milestones at every 50 episodes
    epochs = []
    rewards = []
    avg_losses = []

    if not os.path.exists('ImageWithLabels'):
        os.makedirs('ImageWithLabels')

    for episode in range(num_episodes):
        start_time = time.time()  # Start timer for the episode

        state = env.reset()
        state = np.stack([state] * 4, axis=0)  # Stacking frames
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.stack([next_state] * 4, axis=0)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        rewards.append(total_reward)
        avg_losses.append(np.mean(episode_losses) if episode_losses else 0)

        # Save image with bounding box for each episode
        filename = os.path.join('ImageWithLabels', f'labeled_image_episode_{episode+1}.png')
        env.save_image_with_label(filename)

        end_time = time.time()  # End timer for the episode
        episode_duration = end_time - start_time  # Calculate episode duration
        total_duration += episode_duration  # Update total duration

        if (episode + 1) % 100 == 0:
            # Plot every 100 episodes
            plot_real_time(rewards, avg_losses)
            rewards = []
            avg_losses = []

        if episode + 1 in milestones:  # If this is a milestone episode
            cumulative_time = total_duration / 60  # Convert to minutes
            cumulative_times.append(cumulative_time)
            epochs.append(episode + 1)
            # Save cumulative reward to file
            with open('dqn_reward.txt', 'a') as f:
                f.write(f"Episode {episode + 1} - Reward: {total_reward}\n")
            
            # Save cumulative time to file
            with open('dqn_times.txt', 'a') as f:
                f.write(f"Episode {episode + 1}: {total_duration / 60:.2f} minutes\n")

            # Save cumulative loss to file
            with open('dqn_loss.txt', 'a') as f:
                f.write(f"Episode {episode + 1} - Average Loss: {np.mean(episode_losses):.6f}\n")

            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}, "
                  f"Average Loss: {np.mean(episode_losses):.6f}, Epsilon: {epsilon:.3f}, "
                  f"Cumulative Time: {format_duration(total_duration)}")

            # Plot cumulative time every 50 episodes
            plot_cumulative_time(epochs, cumulative_times)

        # Save the trained model
        if (episode + 1) % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"model_episode_{episode + 1}.pth")

if __name__ == "__main__":
    pygame.init()
    image_folder = "InputImage"
    env = BrainEnv(image_folder)
    state_shape = (4, 60, 60)
    action_size = 5
    agent = DQNAgent(state_shape, action_size)
    summary(agent.policy_net, input_size=(4, 60, 60))  # Print model summary
    train(env, agent)
    pygame.quit()
