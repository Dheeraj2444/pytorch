#!/usr/bin/env python3

import torch
from torch.distributions import normal
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#defining parameters
in_size, out_size = 1, 1
epochs = 50
learning_rate = 0.01

#generate random regression data
dist = normal.Normal(0, 0.5)
x = dist.sample((1000, 1))
dist1 = normal.Normal(0, 0.05)
c = dist1.sample((1000, 1))
y = 0.5 * x + c + 0.1

#define model
model = nn.Linear(in_features=in_size, out_features=out_size)

# #define loss and optimizer
criteria = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#training
for epoch in range(epochs):

	#forward pass
	output = model(x)
	loss = criteria(output, y)

	#backward pass
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if (epoch + 1) % 5 == 0:
		print("Epoch " + str(epoch + 1) + " Loss: " + str(loss.item()))
		
		#visualize prediction
		prediction = model(x)
		y_pred = prediction.data
		plt.scatter(x.numpy(), y.numpy(), color='red')
		plt.plot(x.numpy(), y_pred.numpy())
		plt.show()
		