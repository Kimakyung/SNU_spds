#import the nescessary libs
import numpy as np
import torch
import time

# Loading the Fashion-MNIST dataset
from torchvision import datasets, transforms

start=time.time()

# Get GPU Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                                                   ])
# Download and load the training data
trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = True, num_workers=4)

# Examine a sample
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


# Define the network architecture
from torch import nn, optim
import torch.nn.functional as F

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 10),
                      nn.LogSoftmax(dim = 1)
                     )
model.to(device)

# Define the loss
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Define the epochs
epochs = 5

trainingTime=0
train_losses, test_losses = [], []

for e in range(epochs): 
    Trainingstart = time.time()  # 시작 시간 저장
    running_loss = 0
    for images, labels in trainloader:
        # Flatten Fashion-MNIST images into a 784 long vector
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.shape[0], -1)
        # Training pass
        optimizer.zero_grad()
        output = model.forward(images)
        y_one_hot=torch.zeros_like(output)
        y_one_hot.scatter_(1, labels.view(-1,1),1)
        loss = criterion(output, y_one_hot)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()        
    else:
        test_loss = 0
        accuracy = 0
    trainingTime+=time.time() - Trainingstart

    # Turn off gradients for validation, saves memory and computation
    with torch.no_grad():
      # Set the model to evaluation mode
        model.eval()
      
      # Validation pass
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.shape[0], -1)
            ps = model(images)
            y_one_hot=torch.zeros_like(ps)
            y_one_hot.scatter_(1, labels.view(-1,1),1)
            test_loss += criterion(ps,y_one_hot)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    
    model.train()

    print("Epoch: {}/{}..".format(e+1, epochs),
          "Training loss: {:.3f}..".format(running_loss/len(trainloader)),
          "Test loss: {:.3f}..".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))


EntireTime=time.time()-start    
print("Traning time :", trainingTime)  # 현재시각 - 시작시간 = 실행 시간
print("Entire code excute time:", EntireTime)