import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #((R,G,B layer's average),(R,G,B layer's standard deviation))
    ]
)

#define training data set
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform) #CIFAR10 is a existed traning set
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

#define testing data set
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

#show our data
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    #input data: torch.tensor type and the shape is torch.tensor[c,h,w]
    img = img/2+0.5 #reverse narmalize
    nping = img.numpy()
    nping = np.transpose(nping,(1,2,0)) #turn[c,j,w] to [h,w,c]
    plt.imshow(nping)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self): #define neural network structure, input data 1*32*32
        super(Net, self).__init__()
        #first layer (convolution layer)
        self.conv1 = nn.Conv2d(3,6,3) #input:1, output:6, convolution layer:3*3
        #second layer (convolution layer)
        self.conv2 = nn.Conv2d(6,16,3) #input:6, output:16, convolution layer:3*3
        #third layer (fully connected layer)
        self.fc1 = nn.Linear(16*28*28, 512) #the input dimension 16*28*28(32-2-2 = 28), the output dimension 512
        #fourth layer (fully connected layer)
        self.fc2 = nn.Linear(512,64)
        #fifth layer (fully connected layer)
        self.fc3 = nn. Linear(64,10)

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

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

def main():
    for epoch in range(2):
        for i,data in enumerate(trainloader):
            images,labels = data

            outputs = net(images)

            loss = criterion(outputs,labels) #calculate loss

            optimizer.zero_grad() #before calculate grad,we eneed to clear
            loss.backward()
            optimizer.step()

            if(i%1000 == 0):
                print('Epoch: %, step: %d, Loss: %.3f', %(epoch,i,loss.item()))

if __name__ == '__main__':

    main()
