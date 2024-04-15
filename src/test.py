import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Data preparation
x = torch.linspace(-6, 6, 100).view(-1, 1)
y = torch.sin(x) - 0.1 * x ** 2

# Define the neural network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the network
net = Net()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Training the network
for epoch in range(5000):  # number of epochs
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/5000] Loss: {loss.item():.4f}')
# Save the model weights
torch.save(net.state_dict(), 'model_weights.pth')

# Plotting the results
predicted = net(x).detach().numpy()
plt.scatter(x.numpy(), y.numpy(), label='Actual Data', color='blue')
plt.plot(x.numpy(), predicted, label='Learned Function', color='red')
plt.title('Neural Network Approximation')
plt.legend()
plt.show()
