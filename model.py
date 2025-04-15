import torch 
from torch import nn 
class RPSModule(nn.Module):
    def __init__(self):
        super(RPSModule, self).__init__()
        self.net=nn.Sequential(

            nn.Conv2d(1,32,kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Conv2d(32,64,kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Conv2d(64,64,kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(64*14*14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512,3)
        )

    def forward(self,x ):
        return self.net(x)