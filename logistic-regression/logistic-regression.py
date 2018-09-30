#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader 
import torch.optim as optim

#parameters
batch_size = 32
input_size = 784
nb_classes = 10
learning_rate = 0.001
epochs = 15

#defining sequence of transformations
transform = transforms.Compose([transforms.ToTensor(),
							   transforms.Normalize((0.1307,),(0.3081,))])

#loading MNIST dataset
trainset = torchvision.datasets.MNIST(root='./mnist-data', train=True,\
									 download=True, transform=transform)

#defining data loader to load batches of images
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, \
	num_workers=2)

testset = torchvision.datasets.MNIST(root='./mist-data', train=False,\
									 download=True, transform=transform)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, \
	num_workers=2)

#analyze dataset
train_samples, train_batches = len(trainset), len(trainloader)
test_samples, test_batches = len(testset), len(testloader)
img, label = trainset[0]

print("total number of training samples: ", train_samples)
print("total number of training batches: ", train_batches)
print("total number of testing samples: ", test_samples)
print("total number of testing batches: ", test_batches)
print("input shape of first sample: ", img.size())
print("label of the first sample: ", label)

#define model
model = nn.Linear(in_features=28*28, out_features=nb_classes)

#define error metric and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#model training
for epoch in range(1, epochs + 1):
	for i, (images, labels) in enumerate(trainloader):
		images = images.reshape(-1, 28*28)
		outputs = model(images)
		loss = criteria(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 100 == 0:
			print("Epoch: ", epoch, "batch: ", i + 1, "Loss: ", loss.item())

#model testing
with torch.no_grad():
	total, corrected = 0, 0
	for images, labels in testloader:
		images = images.reshape(-1, 28*28)
		outputs = model(images)
		_, predictions = torch.max(outputs.data, 1)
		total += labels.size()[0]
		corrected += ((predictions==labels).sum()).item()

#printing accuracy
print("Accuracy: ", (corrected / float(total)) * 100) 	
