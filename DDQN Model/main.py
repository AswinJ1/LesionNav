import numpy as np
import torch
from environment import BrainEnv
from agent import DDQNAgent
import pygame
import os
import matplotlib.pyplot as plt
import time
from torchsummary import summary

from model import DQN  # Import the summary function

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

def train(env, agent, num_episodes=1050, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):#epsilon_decay=0.995
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

        end_time = time.time()  # End timer for the episode
        epoch_duration = end_time - start_time  # Duration in seconds
        total_duration += epoch_duration  # Add to total duration

        duration_formatted = format_duration(epoch_duration)
        total_duration_formatted = format_duration(total_duration)
        print(f"Episode: {episode}, Duration: {duration_formatted}, Total Duration: {total_duration_formatted}, Total Reward: {total_reward}, Epsilon: {epsilon}")

        if done:
            env.save_image_with_label(f"ImageWithLabels/episode_{episode}.png")

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        rewards.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_losses.append(avg_loss)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Average Loss: {avg_loss}, Epsilon: {epsilon}")

        if (episode) % 10 == 0:
            agent.update_target_net()

        if (episode) % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"model_episode_{episode}.pth")

        if (episode) % 50 == 0:
            epochs.append(episode)
            cumulative_times.append(total_duration / 60)  # Store total duration in minutes
            
            # Save cumulative time to file
            with open('ddqn_times.txt', 'a') as f:
                f.write(f"Episode {episode}: {total_duration / 60:.2f} minutes\n")
            with open('ddqn_reward.txt', 'a') as f:
                f.write(f"Episode {episode + 1} - Reward: {total_reward}\n")
                
            plot_cumulative_time(epochs, cumulative_times)  # Plot at every milestone

        if (episode) % 100 == 0:
            plot_real_time(rewards, avg_losses)

    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((240, 240))
    pygame.display.set_caption("Brain Tumor Detection")
    env = BrainEnv("InputImage")

    # Instantiate the agent and print the model summary
    state_shape = (4, 60, 60)  # Example input shape (4 stacked frames of 60x60 each)
    num_actions = 5  # Number of actions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_shape, num_actions).to(device)
    
    # Print the model summary
    summary(model, input_size=(state_shape))

    agent = DDQNAgent((4, 60, 60), 5)  # Updated to 5 actions
    train(env, agent)
    pygame.quit()
