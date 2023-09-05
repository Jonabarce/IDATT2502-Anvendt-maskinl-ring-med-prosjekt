import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



inputs = torch.tensor([[0,0], [0,1],[1,0],[1,1]], dtype=torch.float)
targets = torch.tensor([[1],[1],[1],[0]], dtype=torch.float)


class NANDOperator(nn.Module):
    def __init__(self):
        super(NANDOperator, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


model = NANDOperator()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 10000

for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
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
    print(f"NAND(0,0) = {test_00.item()}")
    print(f"NAND(0,1) = {test_01.item()}")
    print(f"NAND(1,0) = {test_10.item()}")
    print(f"NAND(1,1) = {test_11.item()}")

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
fig = plt.figure('Linear regression 3D')
ax = fig.add_subplot(111, projection='3d')

ax.scatter(inputs_np[correct_indices, 0], inputs_np[correct_indices, 1], predicted_outputs[correct_indices], c='green', label='Correct', marker='o')
ax.scatter(inputs_np[incorrect_indices, 0], inputs_np[incorrect_indices, 1], predicted_outputs[incorrect_indices], c='red', label='Incorrect', marker='x')

# Marker de faktiske målverdiene (dette vil gi en bedre forståelse av hvor godt modellen presterer)
ax.scatter(inputs_np[:, 0], inputs_np[:, 1], targets_np[:, 0], c='blue', label='True Value', marker='^')

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')
ax.legend()

plt.show()