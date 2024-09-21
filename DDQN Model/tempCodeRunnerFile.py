from optparse import Values
import numpy as np
import torch
from environment import BrainEnv
from agent import DDQNAgent
import pygame
import os
import matplotlib.pyplot as plt
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
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    
    plt.pause(0.001)
    plt.show()

def train(env, agent, num_episodes=300, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
    epsilon = epsilon_start
    rewards = []
    losses = []

    if not os.path.exists('ImageWithLabels'):
        os.makedirs('ImageWithLabels')

    for episode in range(num_episodes):
        state = env.reset()
        state = np.stack([state] * 4, axis=0)  # Stacking frames
        total_reward = 0
        done = False

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
                losses.append(loss)

        if done:
            env.save_image_with_label(f"ImageWithLabels/episode_{episode}.png")

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        rewards.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Loss: {loss}")

        if episode % 10 == 0:
            agent.update_target_net()

        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f"model_episode_{episode}.pth")

        if episode % 100 == 0:
            plot_real_time(rewards, losses)

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
