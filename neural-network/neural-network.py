#!/usr/bin/env python3

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import MNIST

#parameters
batch = 32
hidden_1 = 100
hidden_2 = 50
learning_rate = 0.001
epochs = 10
input_size = 784
nb_classes = 10

#transformations
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))])

#load dataset
trainset = MNIST(root='./mnist-data/', train=True, download=True, \
				transform=transform)
trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, \
						num_workers=2)

testset = MNIST(root='./mnist-data/', train=False, download=True,\
				transform=transform)
testloader = DataLoader(testset, batch_size=batch, shuffle=False,\
						num_workers=2)

#define model architecture
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_1, hidden_2, nb_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_1)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(in_features=hidden_1, out_features=hidden_2)
		self.fc3 = nn.Linear(in_features=hidden_2, out_features=nb_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		out = self.fc3(x)
		return out

model = NeuralNet(input_size, hidden_1, hidden_2, nb_classes)

#define metric and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#model training
for epoch in range(1, epochs + 1):
	for i, (images, labels) in enumerate(trainloader):
		images = images.view(-1, 28*28)
		outputs = model(images)
		loss = criteria(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 100 == 0:
			print("Epoch: ", epoch, "Batch: ", i + 1, "Loss: ", loss.item())

#evaluate performance
with torch.no_grad():
	total, corrected = 0, 0
	for images, labels in testloader:
		images = images.view(-1, 28*28)
		outputs = model(images)
		_, predictions = torch.max(outputs.data, 1)
		total += labels.size()[0]
		corrected += ((predictions==labels).sum()).item()

#print accuracy
print("Accuracy: ", (corrected / float(total)) * 100)

