import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

inputs = torch.tensor([[0.], [1.]], dtype=torch.float32)
targets = torch.tensor([[1.], [0.]], dtype=torch.float32)

class NOTOperator(nn.Module):
    def __init__(self):
        super(NOTOperator, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


model = NOTOperator()
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
    test_values = torch.linspace(0, 1, 100).reshape(-1, 1)
    predictions = model(test_values)
    test_0 = model(torch.tensor([[0.]]))
    test_1 = model(torch.tensor([[1.]]))
    print(f"NOT(0) = {test_0.item()}")
    print(f"NOT(1) = {test_1.item()}")

# Visualiser modellens prediksjoner
plt.scatter(inputs, targets, color='red', label='True Values')  # Plot de ekte NOT verdiene
plt.plot(test_values, predictions, color='blue', label='Model Predictions') # Plot modellens prediksjoner
plt.xlabel('Input Value')
plt.ylabel('Output Value')
plt.title('NOT Operator Visualization')
plt.legend()
plt.grid(True)
plt.show()