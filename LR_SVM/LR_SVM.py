from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


# Don't change batch size
batch_size = 64
learning_rate = 1

model = nn.Linear(28*28, 1)
criterion = nn.BCEWithLogitsLoss()

while(1):
    choice = input("please input: 1 = LinearRegression with SGD, 2 = LinearRegression with SGD momentum\n"
                   "              2 = SVM  with SGD,             4 = SVM with SGD momentum\n")
    if choice == "2" or choice == "4":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        break
    elif choice == "1" or choice == "3":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        break
    else:
        print("incorrect input, please input again:")
        continue

# USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA
train_data = datasets.MNIST('/data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('/data/mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().reshape(-1)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().reshape(-1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(subset_indices))


# Training
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        # Convert labels from 0,1 to -1,1

        # forward
        outputs = model(images)
        outputs = outputs.squeeze()
        # loss
        if choice == "1" or choice == "2":
            loss = criterion(outputs.float(), labels.float())
        elif choice == "3" or choice == "4":
            labels = Variable(2 * (labels.float() - 0.5))
            loss = torch.mean(torch.clamp(1 - labels * outputs, min=0))
        # loss = criterion(outputs.float(), labels.float())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Test
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    # Put your prediction code here
    prediction = model(images)
    prediction = prediction.data.sign()/2 + 0.5
    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model with SGD on the test images: %f %%' % (100 * (correct.float() / total)))


    
