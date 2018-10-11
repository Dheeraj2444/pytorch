#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from torchvision.models import *

import copy
from tqdm import tqdm
import argparse

def load_data():
	#define transformations
	train_transforms = transforms.Compose([
	        transforms.RandomResizedCrop(224),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470  0.2434  0.2616])
	    ])

	test_transforms = transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470  0.2434  0.2616])
	    ])

	#loading data
	trainset = CIFAR10(root='./cifar10-data/', train=True, download=True, 
					  transform=train_transforms)
	trainloader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)

	testset = CIFAR10(root='./cifar10-data/', train=False, download=True,
					 transform=test_transforms)
	testloader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)

	return trainset, trainloader, testset, testloader

#defining training function
def training(model, criteria, optimizer, epochs):
	best_model = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(1, epochs + 1):
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
				total, total_loss, corrected = 0, 0, 0  
				for i, (images, labels) in enumerate(tqdm(trainloader)):
					inputs = inputs.to(device)
					labels = labels.to(device)
					torch.set_grad_enabled(True)
					outputs = model(images)
					_, predictions = torch.max(outputs.data, 1)
					loss = criteria(outputs, labels)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					total += labels.size()[0]
					total_loss += loss.item()
					corrected += (((predictions==labels)).sum()).item()
				acc = (corrected / float(total)) * 100
				print("Epoch: ", epoch, "total_loss: ", total_loss, \
				"Train Accuracy: ", acc)	
			else:
				model.eval()
				with torch.no_grad():
					total, total_loss, corrected = 0, 0, 0
					for images, labels in tqdm(testloader):
						inputs = inputs.to(device)
						labels = labels.to(device)
						outputs = model(images)
						_, predicted = torch.max(outputs.data, 1)
						corrected += ((predicted==labels).sum()).item()
						total += labels.size()[0]
						total_loss += loss.item()
				acc = (corrected / float(total)) * 100
				if acc > best_acc:
					best_acc = acc
					best_model = copy.deepcopy(model.state_dict())
				print("Loss:", total_loss, "Test Accuracy: ", acc)

	model.load_state_dict(best_model)
	return model

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='CIFAR-10 Transfer Learning')
	parser.add_argument('-model', help='Modelname needed. Available option: \
		alexnet, vgg16, resnet18', required=True)
	parser.add_argument('-finetune', help="Boolean for finetuning trained \
		weights", default=False)
	args = parser.parse_args()

	#define parameters
	batch = 8
	learning_rate = 0.001
	epochs = 50
	nb_classes = 10

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	trainset, trainloader, testset, testloader = load_data()

	if args.model == 'vgg16':
		#convnet as feature extractor
		model = models.vgg16(pretrained=True)
		print(model)

		#freezing convolutional layers (feature extractors)
		if args.finetune == False:
			for param in model.features.parameters():
				param.requires_grad = False

		#changing last layer of classifier
		# num_ftrs = model.classifier[6].in_features
		# features = list(model.classifier.children())[:-1]
		# features.extend([nn.Linear(num_ftrs, nb_classes)])
		# model.classifier = nn.Sequential(*features)
		num_ftrs = model.classifier[-1].in_features
		model.classifier[-1] = nn.Linear(num_ftrs, nb_classes)
		print(model)
		model.to(device)


	if args.model == 'resnet18':
		model = models.resnet18(pretrained=True)
		print(model)

		if args.finetune == False:
			for param in model.parameters():
				param.requires_grad = False

		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, nb_classes)
		model.to(device)


	if args.model == 'alexnet':
		model = models.alexnet(pretrained=True)
		print(model)
		
		if args.finetune == False:
			for param in model.parameters():
				param.requires_grad = False

		# num_ftrs = model.classifier[6].in_features
		# features = list(model.classifier.children())[:-1]
		# features.extend([nn.Linear(num_ftrs, nb_classes)])
		# model.classifier = nn.Sequential(*features)
		num_ftrs = model.classifier[-1].in_features
		model.classifier[-1] = nn.Linear(num_ftrs, nb_classes)
		print(model)
		model.to(device)

	#define metric and optimizer
	criteria = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	model = training(model, criteria, optimizer, epochs)

