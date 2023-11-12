import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
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
path = "./model/MLP.pt"

class MLP_cls(nn.Module):
    def __init__(self,in_dim=3*32*32):
        super(MLP_cls,self).__init__()
        self.lin1 = nn.Linear(in_dim, 32*32)
        self.lin2 = nn.Linear(32*32, 32*16)
        self.lin3 = nn.Linear(32*16, 32*8)
        self.lin4 = nn.Linear(32*8, 32*4)
        self.lin5 = nn.Linear(32*4, 64)
        self.lin6 = nn.Linear(64, 32)
        self.lin7 = nn.Linear(32, 10)

        self.relu = nn.ReLU()
        init.xavier_uniform_(self.lin1.weight)
        init.xavier_uniform_(self.lin2.weight)
        init.xavier_uniform_(self.lin3.weight)
        init.xavier_uniform_(self.lin4.weight)
        init.xavier_uniform_(self.lin5.weight)
        init.xavier_uniform_(self.lin6.weight)
        init.xavier_uniform_(self.lin7.weight)

    def forward(self,x):
        x = x.view(-1,3*32*32)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.relu(x)
        x = self.lin5(x)
        x = self.relu(x)
        x = self.lin6(x)
        x = self.relu(x)
        x = self.lin7(x)
        x = self.relu(x)
        return x
    """
    def forward(self,x):  #without Relu
        x = x.view(-1,3*32*32)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.lin6(x)
        x = self.lin7(x)
        return x

    """

epochs = 10
net = MLP_cls()
#net.load_state_dict(torch.load(path))
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

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
        correct_num  += torch.sum(pred==target)
    print('epoch',epoch,'loss {:.2f}'.format(run_loss.item()/len(train_loader)),'accuracy {:.2f}'.format(correct_num.item()/(len(train_loader)*batch_size)))

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

