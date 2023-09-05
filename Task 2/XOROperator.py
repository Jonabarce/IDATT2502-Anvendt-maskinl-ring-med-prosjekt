import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


inputs = torch.tensor([[0,0], [0,1],[1,0],[1,1]], dtype=torch.float)
targets = torch.tensor([[1],[1],[1],[0]], dtype=torch.float)



class XOROperator(nn.Module):
    def __init__(self):
        super(XOROperator, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(x))



model = XOROperator()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 10000

for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


