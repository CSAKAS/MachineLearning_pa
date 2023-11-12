import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time
import torch.nn.init as init


def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 64

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "./model/CNN.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):  # with Relu
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    """
    def forward(self, x):  # without Relu
        x = F.relu(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    """


net = Net()
#net.load_state_dict(torch.load(path))
net.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

epochs = 10

net.train()
for epoch in range(epochs):
    run_loss = 0
    correct_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        out = net(data)
        _,pred = torch.max(out,dim=1)
        optimizer.zero_grad()
        loss = criterion(out,target)
        loss.backward()
        run_loss += loss
        optimizer.step()
        correct_num += torch.sum(pred==target)
    print('epoch', epoch, 'loss {:.2f}'.format(run_loss.item()/len(train_loader)),'accuracy {:.2f}'.format(correct_num.item()/(len(train_loader)*batch_size)))

torch.save(net.state_dict(), path)

net.eval()
test_loss = 0
test_correct_num = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        out = net(data)
        _,pred = torch.max(out,dim=1)
        test_loss += criterion(out,target)
        test_correct_num  += torch.sum(pred==target)
print('loss {:.2f}'.format(test_loss.item()/len(test_loader)),'accuracy {:.2f}'.format(test_correct_num.item()/(len(test_loader)*test_batch_size)))

