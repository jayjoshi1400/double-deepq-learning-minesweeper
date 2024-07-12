import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from collections import deque
import numpy as np
import random

class DoubleDQN(nn.Module):
    
    def __init__(self, input_size, num_actions):
        super(DoubleDQN, self).__init__()
        
        self.epsilon = 1
        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    ### This is important, masks invalid actions
    def safe_softmax(self,vec, action_mask, dim=1, epsilon=1e-5):
            exps = torch.exp(vec)
            masked_exps = exps * action_mask.float()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)
        
    def forward(self, state, action_mask):
        state = state/8
        state = self.feature(state)
        advantage = self.safe_softmax(self.advantage(state),action_mask)
        value     = self.safe_softmax(self.value(state),action_mask)
        return value + advantage  - advantage.mean()
    
    def select_action(self,state,action_mask):
        random_chance = random.random()
        if random_chance > self.epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0)
            action_mask   = torch.FloatTensor(action_mask).unsqueeze(0)
            q_value = self.forward(state,action_mask)
            action  = q_value.max(1)[1].data[0].item()
        else:
            indices = np.nonzero(action_mask)[0]
            randno = random.randint(0,len(indices)-1)
            action = indices[randno]
        return action


class ReplayBuffer():
    def __init__(self, max_size):
        self.memory = deque(maxlen = max_size)

    def push(self, state, action, action_mask, reward, next_state, next_action_mask, done):
        self.memory.append((state, action, action_mask, reward, next_state, next_action_mask, done))

    def retrieve(self,batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, action_masks, rewards, next_states, next_action_masks, dones = zip(*batch)
        return states, actions, action_masks, rewards, next_states, next_action_masks, dones
