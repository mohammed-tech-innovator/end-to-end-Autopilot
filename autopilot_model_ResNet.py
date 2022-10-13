import torch 
import torch.nn as nn
from torch import flatten
import torch.nn.functional as f 
import torch.optim as optim
import time as t


device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')

class ResNet(nn.Module) :
    def __init__(self , in_channels , out_channels):
        super(ResNet , self).__init__()
        self.conv1 = nn.Conv2d(in_channels , out_channels - in_channels, kernel_size = 3 , stride = 1 , padding = 1)
        self.n1 = nn.InstanceNorm2d(out_channels - in_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels - in_channels, out_channels - in_channels, kernel_size = 3 , stride = 2 , padding = 1)
        self.n2 = nn.InstanceNorm2d(out_channels - in_channels, affine=True)
        self.mish = nn.Mish(inplace=True)

    def forward(self , x):

        identity = x
        x = self.conv1(x)
        x = self.n1(x)
        x = self.mish(x)
        x = self.conv2(x)
        x = self.n2(x)
        x = torch.cat((x,f.max_pool2d(identity,(2,2))),1)
        x = self.mish(x)
        return(x)



class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()

        self.conv = nn.Sequential(
            nn.InstanceNorm2d(3, affine=True),
            nn.Conv2d(3,8,kernel_size = (14,6)) , 
            nn.InstanceNorm2d(8, affine=True),
            nn.Mish(inplace=True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(8,16,kernel_size = (10,5)) , 
            nn.InstanceNorm2d(16, affine=True),
            nn.Mish(inplace=True),
            )

        self.ResNets = nn.Sequential(
            ResNet(16 , 32),
            ResNet(32 , 64),
            ResNet(64 , 128),
            ResNet(128 , 256),
            ResNet(256,512),

            )
        ###########################################
        self.fc = nn.Sequential(
            nn.Linear(1536,768),
            nn.Mish(inplace=True),
            nn.Linear(768,384),
            nn.Mish(inplace=True),
            nn.Linear(384,192),
            nn.Mish(inplace=True),
            nn.Linear(192,4),
            )

    def forward(self,x):

        x = self.conv(x)
        x = self.ResNets(x)
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