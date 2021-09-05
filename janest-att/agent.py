from snet import DQN
from replayMemory import ReplayMemory
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from services import predict,q_estimate,q_target,train_on_batch,DEVICE

class Agent:

    def __init__(self,
    state_size,action_size,
    batch_size=128,gamma=0.999,
    EPS=1.0,EPS_MIN=0.05,
    EPS_DECAY=0.990,memory_size=10000,
    target_sync=1024, learn_every=1,
    checkpoint=None) -> None:
        
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.EPS = EPS
        self.EPS_MIN = EPS_MIN
        self.EPS_DECAY = EPS_DECAY
        self.memory_size = memory_size
        self.target_sync = target_sync
        self.learn_every = learn_every
        self.checkpoint = checkpoint

        self.memory = ReplayMemory(self.state_size,self.memory_size)

        self.q_net = DQN(state_size,action_size).to(DEVICE)
        self.target_net = DQN(state_size,action_size).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.q_net.parameters())
        self.lossFn = nn.SmoothL1Loss()

        self.steps = 0

        if self.checkpoint:
            self.load_checkpoint(checkpoint)

    def act(self, state):

        action = None
        if np.random.rand() <= self.EPS:
            action = np.random.choice(self.action_size)
            
        else:
            act_values = predict(self.q_net, state)
            action = np.argmax(act_values[0])
            

        #EPS Decay
        if self.EPS > self.EPS_MIN:
            self.EPS *= self.EPS_DECAY

        self.steps+=1

        return action


    def learn(self):

        if(self.steps < self.batch_size):
            return None, None 

        if(self.steps % self.target_sync == 0):
            self.syncTarget()
        
        if(self.steps % self.learn_every != 0):
            return None, None     

        state, next_state, action, reward, done = self.recall()
        
        q_est = q_estimate(self.q_net,state, action)

        
        q_tgt = q_target(self.target_net,self.q_net,self.gamma,reward, next_state, done)

       
        loss = train_on_batch(q_est, q_tgt, self.lossFn, self.optimizer)

        return (q_est.mean().item(), loss)
        

    def cache(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def recall(self):
        return self.memory.sample(batch_size=self.batch_size)

    def syncTarget(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path,map_location=DEVICE)

        self.q_net.load_state_dict(checkpoint['model_state_dict'])
        self.syncTarget()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.EPS = checkpoint['eps']     