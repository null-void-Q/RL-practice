import numpy as np

#TODO should include date in state ?

class ENV:
    def __init__(self, data, response, dates = None,weights = None ,n_actions=2) -> None:
        # data should be np arr of [weight,features...]
        # response should be np arr of [resp...]

        self.data = data
        self.response = response
        self.dates = dates
        self.weights = weights

        self.n_steps = data.shape[0]
        self.curr_step = 0

        self.state_size = data.shape[1]


        self.n_actions = n_actions
        self.action_space = np.arange(self.n_actions)


        self.curr_score = 0


    def step(self,action):
        assert action in self.action_space

        # action(0 or 1 ) * response * weight
        if action == 1:
            reward = self.response[self.curr_step]
        else:
            reward = -1 * self.response[self.curr_step] 
        #move step to get next state
        self.curr_step +=1
        
        next_state = self.data[self.curr_step]

        #done when data is over
        done = self.curr_step == self.n_steps - 1

        self.curr_score += reward

        #populate with informative valuess
        info = {'curr_score': self.curr_score, 'Current Step':self.curr_step}



        return next_state, reward, done, info


    def reset(self):

        self.curr_step = 0
        self.curr_score = 0

        return self.data[self.curr_step]