from snet import DQN
from data import loadAndPreprocess
import torch
from services import DEVICE,predict,calc_utility_score
import numpy as np

checkpoint_path = './checkpoints/2021-02-14T10-37-18/s_dqn_5.chkpt'
data_path = '../data/b_validation.csv'


def test(model_path,data_path):

    checkpoint = torch.load(model_path,map_location=DEVICE)
    q_net = DQN(130,2).to(DEVICE)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    q_net.eval()

    features,response,dates,weights = loadAndPreprocess(data_path)
    actions = np.zeros(len(response), dtype=np.uint8)

    for i,feature in enumerate(features):
        print('* playing: ',i,'/',len(features), end='\r')
        res = predict(q_net,feature)
        actions[i] = np.argmax(res[0])
        
    print()
    print(calc_utility_score(dates,weights,response,actions))
    print(np.count_nonzero(actions),np.count_nonzero(actions)/len(actions))    


if __name__ == '__main__':
    test(checkpoint_path,data_path)

