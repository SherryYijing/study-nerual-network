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

#in Windows environment need to have these code
if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')

    dataIter = iter(trainloader) #random loading a mini batch

    images,labels = dataIter.next()

    imshow(torchvision.utils.make_grid(images))
    
    plt.show()
