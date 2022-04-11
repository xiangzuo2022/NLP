from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision import utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
cwd = os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#add CIFAR10 data in the environment
sys.path.append(cwd + 'cifar10')

#Numpy is linear algebra lbrary
# Matplotlib is a visualizations library
#CIFAR10 is a custom Dataloader that loads a subset ofthe data from a local folder
batch_size = 4


def show_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_data():

    #convert the images to tensor and normalized them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR10(root='cifar10',  transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)
    return trainloader

#1. DEFINE THE CNN HERE
"""
A conv layer with 3 channels as input, 6 channels as output, and a 5x5 kernel
A 2x2 max-pooling layer
A conv layer with 6 channels as input, 16 channels as output, and a 5x5 kernel
A linear layer with 1655 nodes
A linear layer with 120 nodes
A linear layer with 84 nodes
A linear layer with 10 nodes
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#2. TRAIN THE MODEL HERE
def train(model, training_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0

    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(training_data, 0):
            # get the inputs; cifar10 is a list of [inputs, labels]
            inputs, label = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def evaluate(model):
    dataiter = iter(load_data())
    images, labels = dataiter.next()

    # print images
    show_image(utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
            classes[labels[j]] for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                    for j in range(4)))


def main():
    training_data = load_data()
    model = CNN()
    train(model, training_data)
    evaluate(model)

main()
