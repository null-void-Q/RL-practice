import torch.nn as nn


class DQN (nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN,self).__init__()
        
        self.normalize = nn.BatchNorm1d(input_size)
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size, 160)


        self.batchnorm2 = nn.BatchNorm1d(160)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(160, 160)

        self.batchnorm3 = nn.BatchNorm1d(160)
        self.drop3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(160, 160)

        self.batchnorm4 = nn.BatchNorm1d(160)
        self.drop4 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(160, 160)



        self.batchnorm5 = nn.BatchNorm1d(160)
        self.drop5= nn.Dropout(p=0.2)

        self.fc5 = nn.Linear(160, output_size)
    

        self.silu = nn.ReLU()




        

    def forward(self, x):
        x = self.normalize(x)
        x = self.drop1(x)
        x = self.silu(self.fc1(x))
        x = self.batchnorm2(x)
        x = self.drop2(x)
        x = self.silu(self.fc2(x))
        x = self.batchnorm3(x)
        x = self.drop3(x)
        x = self.silu(self.fc3(x))
        x = self.batchnorm4(x)
        x = self.drop4(x)
        x = self.silu(self.fc4(x))
        x = self.batchnorm5(x)
        x = self.drop5(x)
        x = self.fc5(x)
        return x
