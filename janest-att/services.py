import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def predict(model, inputs):
    model.eval()  
    output = model(torch.tensor([inputs],device=DEVICE))
    model.train()
    return output.detach().to('cpu').numpy()



def train_on_batch(td_estimate, td_target,lossFn,optimizer):

    loss = lossFn(td_estimate, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def q_estimate(q_net, state, action):
    state = torch.tensor(state,device=DEVICE)
    action = torch.tensor(action,dtype=torch.long,device=DEVICE)
    current_Q = torch.index_select(q_net(state), dim=1, index=action)  # Q_online(s,a)
    return current_Q

@torch.no_grad()
def q_target(target_net, q_net, gamma,reward, next_state, done):
    
    next_state = torch.tensor(next_state,device=DEVICE)
    reward = torch.from_numpy(reward).to(DEVICE)
    done = torch.from_numpy(done).to(DEVICE)

    next_state_Q = q_net(next_state)
    best_action = torch.argmax(next_state_Q, axis=1)

    next_Q = torch.index_select(target_net(next_state), dim=1, index=best_action)

    return (reward + (1 - done) * gamma * next_Q)


def save_checkpoint(model,optimizer,eps,episode,save_dir):
    path = save_dir / f"s_dqn_{episode}.chkpt"
    torch.save({
        'epoch': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eps': eps,
        },path)

def save(model,path):
    torch.save(model.state_dict(), path)  

def load(model,model_path):
    model.load_state_dict(torch.load(model_path))


def calc_utility_score(date, weight, resp, action):
    count_i = len(pd.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u


def make_scaler(data,scaler_path='./scaler.pkl'):
    if(os.path.exists(scaler_path)):
        scaler = joblib.load(scaler_path)
        return scaler.transform(data)
    scaler = StandardScaler()
    scaler.fit(data)
    joblib.dump(scaler,scaler_path)
    return scaler.transform(data)
