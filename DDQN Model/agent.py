import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from replay_memory import ReplayMemory
from model import DQN

class DDQNAgent:
    def __init__(self, state_shape, action_size, batch_size=128, gamma=0.99, lr=1e-4, memory_size=15000): #lr=1e-4
        self.state_shape = state_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.memory = ReplayMemory(memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_shape, action_size).to(self.device)
        self.target_net = DQN(state_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return np.argmax(q_values.cpu().data.numpy())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

