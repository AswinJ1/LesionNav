
import numpy as np
import torch
from environment import BrainEnv
from agent import DDQNAgent
import pygame
import os
import matplotlib.pyplot as plt
from torchsummary import summary
from model import DQN  # Import the DQN model
import torch.nn.functional as F

def test_plot_real_time(rewards, losses):
    plt.figure(figsize=(10, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Episode Rewards')

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Test Episode Loss')

    plt.tight_layout()
    plt.show()

def test(env, agent, model_path, num_episodes=100):
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    
    rewards = []
    losses = []

    if not os.path.exists('TestImageWithLabels'):
        os.makedirs('TestImageWithLabels')

    for episode in range(num_episodes):
        state = env.reset()
        state = np.stack([state] * 4, axis=0)  # Stacking frames
        total_reward = 0
        total_loss = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.select_action(state, epsilon=0.1)  # Use greedy action selection (epsilon=0.01)
            next_state, reward, done, _ = env.step(action)
            next_state = np.stack([next_state] * 4, axis=0)
            
            # Compute loss
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
            reward_tensor = torch.FloatTensor([reward]).to(agent.device)
            action_tensor = torch.LongTensor([action]).to(agent.device)
            done_tensor = torch.FloatTensor([done]).to(agent.device)

            q_values = agent.policy_net(state_tensor)
            next_q_values = agent.target_net(next_state_tensor)
            q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward_tensor + agent.gamma * next_q_value * (1 - done_tensor)
            loss = F.mse_loss(q_value, expected_q_value)

            total_loss += loss.item()
            state = next_state
            total_reward += reward
            env.render()  # Optionally render the environment

            if done:
                env.save_image_with_label(f"TestImageWithLabels/episode_{episode}.png")

        rewards.append(total_reward)
        losses.append(total_loss)
        print(f"Episode: {episode}, Total Reward: {total_reward}, Loss: {total_loss}")

    # Plot the test rewards and losses after all episodes are done
    test_plot_real_time(rewards, losses)

    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((240, 240))
    pygame.display.set_caption("Brain Tumor Detection")
    env = BrainEnv("TestInputImage")

    # Instantiate the agent and print the model summary
    state_shape = (4, 60, 60)  # Example input shape (4 stacked frames of 60x60 each)
    num_actions = 5  # Number of actions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_shape, num_actions).to(device)
    
    # Print the model summary
    summary(model, input_size=(state_shape))

    agent = DDQNAgent(state_shape, num_actions)
    model_path = "model_episode_850.pth"  # Provide the path to the saved model checkpoint
    test(env, agent, model_path)
    pygame.quit()
