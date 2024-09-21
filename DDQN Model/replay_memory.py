import random
import collections

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
        self.Transition = collections.namedtuple('Transition', 
            ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
