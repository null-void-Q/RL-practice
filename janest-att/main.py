from data import loadAndPreprocess
from market_env import ENV
from agent import Agent
from logger import MetricLogger
from services import save_checkpoint
from pathlib import Path
import datetime

BATCH_SIZE = 6144
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 0.999999
MEMORY_SIZE = 40000
TARGET_SYNC = 5000
LEARN_EVERY = 100
EPISODES = 10

SAVE_EVERY = 1
RECORD_EVERY = 1

checkpoint_path = None

train_file = '../data/x_train.csv'
validation_file = '../data/x_validation.csv'

def main():

    #setup logging 
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    logger = MetricLogger(save_dir)


    features,response,dates, weights = loadAndPreprocess(train_file)
    
    env = ENV(features,response,dates,weights,n_actions = 2)
    env.reset()


    agent = Agent(env.state_size,env.n_actions,
                    batch_size=BATCH_SIZE,gamma=GAMMA,
                    EPS=EPS_START,EPS_MIN=EPS_END,
                    EPS_DECAY=EPS_DECAY,memory_size=MEMORY_SIZE,
                    target_sync=TARGET_SYNC,learn_every=LEARN_EVERY,checkpoint = checkpoint_path)

    for i in range(EPISODES):
        # Initialize the environment and state
        state = env.reset()

        while True:

            # act
            action = agent.act(state)

            # update env
            next_state, reward, done, info = env.step(action)

            # remember
            agent.cache(state, action, reward, next_state, done)

            # learn
            q,loss=agent.learn()

            # log
            logger.log_step(reward, loss, q)

            # Move to the next state
            state = next_state
            
            if done:
                break

        logger.log_episode()
        if i % RECORD_EVERY == 0:
            logger.record(episode=i, epsilon=agent.EPS, step=agent.steps)
                    
        if i % SAVE_EVERY == 0:
            save_checkpoint(agent.q_net,agent.optimizer,agent.EPS,i,save_dir)



if __name__ == '__main__':
    main()
