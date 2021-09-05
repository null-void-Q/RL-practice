import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd

data_path = '../data/b_validation.csv'

class JanestEnv(gym.Env):
  def __init__(self) -> None:
      # data should be np arr of [weight,features...]
      # response should be np arr of [resp...]

      print('Initiating ENV...')

      self.data, self.response,self.dates, self.weights = self.loadData(data_path) 

      self.n_steps = self.data.shape[0]
      self.curr_step = 0

      self.state_size = self.data.shape[1]
      
      self.action_space = spaces.Discrete(2)
      self.observation_space = spaces.Box(low=np.min(self.data,axis=0), high=np.max(self.data,axis=0), dtype=self.data.dtype)

      self.curr_score = 0


  def step(self,action):
      assert action in self.action_space

      # action(0 or 1 ) * response * weight
      reward = self.response[self.curr_step]
      if action == 0:
          reward *= -1

      #move step to get next obsv
      self.curr_step +=1
      
      next_obsv = self.data[self.curr_step]

      #done when data is over
      done = self.curr_step == self.n_steps - 1

      self.curr_score += reward

      #populate with informative valuess
      info = {'curr_score': self.curr_score, 'Current Step':self.curr_step}



      return next_obsv, reward, done, info


  def reset(self):

      self.curr_step = 0
      self.curr_score = 0

      return self.data[self.curr_step]

  def render(self, mode='human'):
    print('ENV RENDER')

  def close(self):
    self.reset()


  def loadData(self,file_path, means_path='./f_mean.npy'):
    print('Loading Data...')

    data = pd.read_csv(file_path,dtype='float32')

    # data reduction
   
    data = data.query('weight > 0').reset_index(drop=True)
    data = data.query('date > 85').reset_index(drop = True)
    

    #feature preprocessing
    features = data.columns[data.columns.str.contains('feature')]
    means = pd.Series(np.load(means_path),index=features[1:],dtype='float32')
    data = data.fillna(means)

    weights = data['weight'].to_numpy()
    features = data[features].to_numpy()
    response = data['resp'].to_numpy()
    dates = data['date'].astype(int).to_numpy()

    print('Scalling Data...')
    features = make_scaler(features)

    print('Data Ready.')
    return features, response, dates, weights

def make_scaler(data,scaler_path='./scaler.pkl'):

    import os
    from sklearn.preprocessing import StandardScaler
    import joblib 

    if(os.path.exists(scaler_path)):
        scaler = joblib.load(scaler_path)
        return scaler.transform(data)
    scaler = StandardScaler()
    scaler.fit(data)
    joblib.dump(scaler,scaler_path)
    return scaler.transform(data)
