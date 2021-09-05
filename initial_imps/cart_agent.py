from nullnet import DQN
from replayMemory import ReplayMemory
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from services import train_on_batch,predict,DEVICE

class Agent:
    def __init__(self,
    state_size,action_size,
    batch_size=128,gamma=0.999,
    EPS=1.0,EPS_MIN=0.05,
    EPS_DECAY=0.990,memory_size = 10000) -> None:
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.EPS = EPS
        self.EPS_MIN = EPS_MIN
        self.EPS_DECAY = EPS_DECAY
        self.memory_size = memory_size

        self.memory = ReplayMemory(self.memory_size)

        self.q_net = DQN(state_size[0],state_size[1],action_size).to(DEVICE)
        self.target_net = DQN(state_size[0],state_size[1],action_size).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.q_net.parameters())
        self.lossFn = nn.SmoothL1Loss()

    def act(self, state):

        if np.random.rand() <= self.EPS:
            return np.random.choice(self.action_size)

        act_values = predict(self.q_net, state)
        return np.argmax(act_values[0])

    def learn(self):
        
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

    

        
        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
  



        # Compute Huber loss
        loss = train_on_batch(self.q_net,self.lossFn,self.optimizer,state_batch, expected_state_action_values.unsqueeze(1),action_batch)

        self.eps_decay()
        
        return loss
        

    def update_memory(self,state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def eps_decay(self):
        if self.EPS > self.EPS_MIN:
            self.EPS *= self.EPS_DECAY
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())                 