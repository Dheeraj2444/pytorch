#!/usr/bin/env python3

#Accuracy: 98.97%

import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

#parameters
batch = 32
kernel1 = 32
kernel1_size = 5
kernel2 = 32
kernel2_size = 5
nb_classes = 10
learning_rate = 0.01
epochs = 10

#load dataset
transforms = transforms.Compose([transforms.ToTensor(),
								 transforms.Normalize((0.1307,), (0.3801,))])

trainset = MNIST(root='./mnist-data/', train=True, download=True,
				transform=transforms)
trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, \
						num_workers=2)

testset = MNIST(root='./mnist-data', train=False, download=True,
				transform=transforms)
testloader = DataLoader(testset, batch_size=batch, shuffle=False, \
					 	num_workers=2)

#define model architecture
class cnn(nn.Module):
	def __init__(self):
		super(cnn, self).__init__()
		self.block1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=kernel1,\
											kernel_size=kernel1_size, padding=2,
											stride=1),
									nn.BatchNorm2d(num_features=kernel1),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2, stride=2))

		self.block2 = nn.Sequential(nn.Conv2d(in_channels=kernel1, out_channels=\
											  kernel2, kernel_size=kernel2_size,
											  padding=0, stride=1),
									nn.BatchNorm2d(num_features=kernel2),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2, stride=2))

		self.fc1 = nn.Linear(in_features=kernel2*5*5, out_features= nb_classes)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = x.view(-1, kernel2*5*5)
		out = self.fc1(x)
		return out

model = cnn()

#define metric and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#model training
for epoch in range(1, epochs + 1):
	for i, (images, labels) in enumerate(trainloader):
		outputs = model(images)
		loss = criteria(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 10 == 0:
			print("Epoch: ", epoch, "Batch: ", i + 1, "Loss: ", loss.item())

#test performance
with torch.no_grad():
	total, corrected = 0, 0
	for images, labels in testloader:
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size()[0]
		corrected += ((predicted == labels).sum()).item()

print("Accuracy: ", (corrected / float(total)) * 100)
