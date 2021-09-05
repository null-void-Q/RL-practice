import gym
import matplotlib.pyplot as plt
from itertools import count
import torch
from cart_utils import get_screen
from cart_agent import Agent
from services import DEVICE

plt.ion()

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 5
MEMORY_SIZE = 10000

EPISODES = 80

def main():

    env = gym.make('CartPole-v0')
    env.reset()

    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    agent = Agent((screen_height, screen_width),n_actions,
                    batch_size=BATCH_SIZE,gamma=GAMMA,
                    EPS=EPS_START,EPS_MIN=EPS_END,
                    EPS_DECAY=EPS_DECAY,memory_size=MEMORY_SIZE)

    for i_episode in range(EPISODES):
        # Initialize the environment and state
        env.reset()

        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen

        for t in count():
            # Select and perform an action
            action = agent.act(state)

            _, reward, done, _ = env.step(action)
            reward = torch.tensor([reward], device=DEVICE)
            action = torch.tensor(action, device=DEVICE).view(1,1)
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            agent.update_memory(state,action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.learn()

            #plot visualization
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target()

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    main()
