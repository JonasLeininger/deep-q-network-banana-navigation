import os
import numpy as np
import torch
import torch.nn.functional as F

from pytorch.dqn import DQN
from pytorch.replay_memory import ReplayMemory


class Agent():

    def __init__(self, state_size: int, action_size: int):
        self.buffer_size = int(1e5)
        self.batch_size = 64
        self.tau = 1e-3
        self.update_every = 4
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.checkpoint_path = "checkpoints/pytorch/cp-{epoch:04d}.pt"
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.qnetwork = DQN(self.state_size, self.action_size)
        self.qnetwork.to(self.device)
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=self.learning_rate)

        self.tragetnetwork = DQN(self.state_size, self.action_size)
        self.tragetnetwork.to(self.device)
        self.tragetnetwork.load_state_dict(self.qnetwork.state_dict())
        
        self.memory = ReplayMemory(self.action_size, self.buffer_size, self.batch_size)
    
    def step(self, state, action, reward, next_state, done):
        '''
        Save experience in replay memory
        
        Params
        :param state: state
        :param action: action
        :param reward: reward
        :param next_state: next_staet
        :param done: done
        :return:
        '''
        self.memory.add(state, action, reward, next_state, done)
    
    def act(self, state):
        '''
        Agents choosen action for given state
        
        Params
        :param state: state to act on
        :return: action
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.qnetwork(state)
        return np.argmax(act_values.cpu().data.numpy())
    
    def replay(self):
        '''
        Train agent on the replay memory. DQN algorithm with local q-network and target-network

        Params
        :return: None
        '''
        states, actions, rewards, next_states, dones = self.memory.sample()
        targetnetwork_outputs = self.tragetnetwork(next_states).max(dim=1)[0].unsqueeze(1)
        targets = rewards + (self.gamma * targetnetwork_outputs)*(1 - dones)
        expected = self.qnetwork(states).gather(1, actions)
        predicts = self.qnetwork(states)
        
        self.loss = F.mse_loss(expected, targets)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.update_target_network()
    
    def update_target_network(self):
        self.tragetnetwork.load_state_dict(self.qnetwork.state_dict())
    
    def save_checkpoint(self, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.qnetwork.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, self.checkpoint_path.format(epoch=epoch))
