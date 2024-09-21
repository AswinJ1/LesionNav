import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(self._conv_output_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _conv_output_size(self, input_shape):
        o = input_shape[1]
        for _ in range(4):
            o = (o + 1) // 2
        return o * o * 32

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        return self.fc2(x)
