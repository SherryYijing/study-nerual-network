import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self): #define neural network structure, input data 1*32*32
        super(Net, self).__init__()
        #first layer (convolution layer)
        self.conv1 = nn.Conv2d(1,6,3) #input:1, output:6, convolution layer:3*3
        #second layer (convolution layer)
        self.conv2 = nn.Conv2d(6,16,3) #input:6, output:16, convolution layer:3*3
        #third layer (fully connected layer)
        self.fc1 = nn.Linear(16*28*28, 512) #the input dimension 16*28*28(32-2-2 = 28), the output dimension 512
        #fourth layer (fully connected layer)
        self.fc2 = nn.Linear(512,64)
        #fifth layer (fully connected layer)
        self.fc3 = nn. Linear(64,2)

    def forward(self,x): #define data flow
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(-1,16*28*28)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x

net = Net()
print(net)

#build a random input
input_data = torch.randn(1,1,32,32) 
print(input_data)
print(input_data.size())

#run nerual network
out = net(input_data)
print(out)
print(out.size())

#build a random real numbuer
target = torch.randn(2)
target = target.view(1,-1)
print(target)

#define loss function and calculate loss
criterion = nn.L1Loss()
loss = criterion(out,target)
print(loss)
