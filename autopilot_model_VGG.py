import torch 
import torch.nn as nn
from torch import flatten
import torch.nn.functional as f 
import torch.optim as optim
import time as t

device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(3, affine=True),
            nn.Conv2d(3,32,5,stride= (2,2)) , 
            nn.InstanceNorm2d(32, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(32,36,5,stride= (2,2)),
            nn.InstanceNorm2d(36, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(36,48,5,stride= (2,2)),
            nn.InstanceNorm2d(48, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(48,56,3),
            nn.InstanceNorm2d(56, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(56,64,3),
            nn.InstanceNorm2d(64, affine=True),
            nn.Mish(inplace=True),
            nn.Conv2d(64,72,3),
            nn.InstanceNorm2d(72, affine=True),
            nn.Mish(inplace=True),
            )
        ###########################################
        self.fc = nn.Sequential(
            nn.Linear(1368,850),
            nn.Mish(inplace=True),
            nn.Linear(850,500),
            nn.Mish(inplace=True),
            nn.Linear(500,250),
            nn.Mish(inplace=True),
            nn.Linear(250,120),
            nn.Mish(inplace=True),
            nn.Linear(120,4),
            )

    def forward(self,x):

        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x





def train(nuralnet , batch  , optimizer , weight = torch.tensor([0.25 ,0.25 ,0.25 ,0.25])):
    
    lossFunction = nn.CrossEntropyLoss(weight = weight)
    x,y = batch
    x = x.to(device)
    y = y.to(device)
    nuralnet.zero_grad()
    output = nuralnet.forward(x)
    loss = lossFunction(output , y)
    loss.backward()
    optimizer.step()

    return nuralnet,loss

def test (nuralnet , batch ,  weight = torch.tensor([0.25 ,0.25 ,0.25 ,0.25])) :

    lossFunction = nn.CrossEntropyLoss(weight = weight)
    x,y = batch
    x = x.to(device)
    y = y.to(device)
    output = nuralnet.forward(x)
    loss = lossFunction(output , y)


    return loss
    
#n = NeuralNet()
#print(n(torch.tensor(torch.rand(1,3,224,78),dtype = torch.float16)))

#print(sum( p.numel() for p in n.parameters() if p.requires_grad))


