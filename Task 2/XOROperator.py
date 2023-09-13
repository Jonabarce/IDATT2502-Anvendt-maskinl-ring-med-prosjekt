import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.nn import init

inputs = torch.tensor([[0,0], [0,1],[1,0],[1,1]], dtype=torch.float)
targets = torch.tensor([[0],[1],[1],[0]], dtype=torch.float)

class XOROperator(nn.Module):
    def __init__(self):
        super(XOROperator, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


    # Without initialization with random values, all nodes in a layer can learn exactly the same way,
    # especially if they start with the same weights.
    # This can cause many nodes in a layer to become redundant, because they learn the same functions.
    def initialize_weights(self):

        init.uniform_(self.fc1.weight, -1, 1)
        init.uniform_(self.fc1.bias, -1, 1)


        init.uniform_(self.fc2.weight, -1, 1)
        init.uniform_(self.fc2.bias, -1, 1)

model = XOROperator()
model.initialize_weights()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 10000

losses = []

for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    test_values = torch.tensor([[0,0], [0,1],[1,0],[1,1]], dtype=torch.float)
    predictions = model(test_values)
    test_00 = model(torch.tensor([[0,0]], dtype=torch.float))
    test_01 = model(torch.tensor([[0,1]], dtype=torch.float))
    test_10 = model(torch.tensor([[1,0]], dtype=torch.float))
    test_11 = model(torch.tensor([[1,1]], dtype=torch.float))
    print(f"XOR(0,0) = {test_00.item()}")
    print(f"XOR(0,1) = {test_01.item()}")
    print(f"XOR(1,0) = {test_10.item()}")
    print(f"XOR(1,1) = {test_11.item()}")


# Visualiser modellens prediksjoner
fig = plt.figure('Linear regression 3D')
ax = fig.add_subplot(111, projection='3d')

# Plot de ekte XOR verdiene
ax.scatter(inputs[:, 0], inputs[:, 1], targets, color='red', label='True Values')

# Plot modellens prediksjoner

# Få numpy-versjon av dataene for visualisering

inputs_np = inputs.numpy()
targets_np = targets.numpy()
predicted_outputs = outputs.detach().numpy()

# Forutsagte klasser basert på 0.5 terskel

predicted_classes = (predicted_outputs > 0.5).astype(int)

# Finn riktige og feil forutsagte indekser

correct_indices = (predicted_classes == targets_np).squeeze()
incorrect_indices = ~correct_indices

# Visualiser

ax.scatter(inputs_np[correct_indices, 0], inputs_np[correct_indices, 1], predicted_outputs[correct_indices], c='green', label='Correct', marker='o')
ax.scatter(inputs_np[incorrect_indices, 0], inputs_np[incorrect_indices, 1], predicted_outputs[incorrect_indices], c='red', label='Incorrect', marker='x')

# Definer ett plan som går igjennom punktene
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = model(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float)).item()

ax.plot_surface(X, Y, Z, alpha=0.5)

# Vis grafen


ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
ax.set_title('XOR Operator Visualization')
ax.legend()
plt.show()
