import numpy as np

class ReplayMemory:

  def __init__(self, state_dim, size):

    self.state_buf = np.zeros([size, state_dim], dtype=np.float32)
    self.next_state_buf = np.zeros([size, state_dim], dtype=np.float32)
    self.actions_buf = np.zeros(size, dtype=np.uint8)
    self.rewards_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.uint8)

    self.ptr, self.size, self.max_size = 0, 0, size

  def push(self, state, act, rew, next_state, done):

    self.state_buf[self.ptr] = state
    self.next_state_buf[self.ptr] = next_state
    self.actions_buf[self.ptr] = act
    self.rewards_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done

    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample(self, batch_size=128):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return (self.state_buf[idxs],
                self.next_state_buf[idxs],
                self.actions_buf[idxs],
                self.rewards_buf[idxs],
                self.done_buf[idxs])

  
  def __len__(self):
        return self.size               

