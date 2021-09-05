import torch
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, inputs):
  with torch.no_grad():
    output = model(inputs)
    return output.detach().to('cpu').numpy()

def train_on_batch(model, criterion, optimizer, inputs, targets,a):
  # convert to tensors
  #inputs = torch.from_numpy(inputs.astype(np.float32)).to(DEVICE)
  #targets = torch.from_numpy(targets.astype(np.float32)).to(DEVICE)

  # zero the parameter gradients
  optimizer.zero_grad()

  # Forward pass
  outputs = model(inputs).gather(1, a)

  loss = criterion(outputs, targets)
        
  # Backward and optimize
  loss.backward()
  optimizer.step()

  return loss