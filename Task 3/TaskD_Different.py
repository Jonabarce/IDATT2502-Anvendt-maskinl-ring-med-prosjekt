import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

print(torch.cuda.is_available())
# Transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Load observations from the Fashion MNIST dataset
train_data = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense1 = nn.Linear(64 * 7 * 7, 1024)
        self.dropout = nn.Dropout(p=0.25)
        self.dense2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.pool1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.pool2(x))
        x = self.dropout(x)
        x = torch.relu(self.dense1(x.reshape(-1, 64 * 7 * 7)))
        return self.dense2(x)

    def loss(self, x, y):
        return nn.functional.cross_entropy(x, y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(x.argmax(1), y).float())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ( "Now training on", device)

model = ConvolutionalNeuralNetworkModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.001)

for epoch in range(20):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        y_pred = model(x_batch)
        loss = model.loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)

            y_pred = model(x_test)
            correct += (y_pred.argmax(1) == y_test).sum().item()
            total += y_test.size(0)

    print(f"Epoch {epoch+1}, Accuracy: {correct / total:.4f}")

print("Training complete!")





