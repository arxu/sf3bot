import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        
        # self.layer1 = nn.Linear(1, in_dim, 64)
        # self.layer2 = nn.Linear(64, 64)
        # self.layer3 = nn.Linear(64, 64)
        # self.layer4 = nn.Linear(64, out_dim)   
        
        # self.layer1 = nn.Conv2d(in_dim, 128, kernel_size=1, stride=1, padding=1)
        # self.layer2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=1)
        # self.layer3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=1)
        # self.layer4 = nn.Conv2d(128, out_dim, kernel_size=1, stride=1, padding=1)
        
        # self.layer1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=4, padding=1)
        # self.layer2 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=4, padding=1)
        # self.layer3 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=4, padding=1)
        # self.layer4 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=4, padding=1)
        
        self.layerx = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Linear(512, out_dim)
        )
        
    #network forward pass taking observation as parameter
    def forward(self, obs):
        #if np array, convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype = torch.float)
        
        #obs.unsqueeze(3)
        
        #activate with ReLU
        # activation1 = F.relu(self.layer1(obs))
        # activation2 = F.relu(self.layer2(activation1))
        # activation3 = F.relu(self.layer3(activation2))
        # output = self.layer4(activation3)
        
        output = self.layerx(obs)
        
        return output

