import gym
from stable_baselines3 import PPO,A2C
from services import calc_utility_score
import numpy as np

env = gym.make('gym_null:janest-v0')

model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=(env.n_steps * 5))
# model.save("./janst_a2c")
model.load("./janst_a2c")
actions = np.zeros(env.n_steps, dtype=np.uint8)
obs = env.reset()
for i in range(env.n_steps):
    action, _states = model.predict(obs, deterministic=True)
    
    actions[i] = action
    obs, reward, done, info = env.step(action)
    print('Predicting: ',i,'/',env.n_steps,end='\r')
    if done:
      obs = env.reset()
print()
print(calc_utility_score(env.dates,env.weights,env.response,actions))
print('1% : ',np.count_nonzero(actions)/len(actions))
env.close()