import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch  # import Patch

# Load data from the CSV file
data = pd.read_csv('day_length_weight.csv', header=0)
print(data.columns)

x_train = torch.tensor(data['# day'].values, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(data['length'].values, dtype=torch.float32).view(-1, 1)
z_train = torch.tensor(data['weight'].values, dtype=torch.float32).view(-1, 1)

# Store means and stds for denormalization
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
z_mean, z_std = z_train.mean(), z_train.std()

# Normalize the data
x_train_normalized = (x_train - x_mean) / x_std
y_train_normalized = (y_train - y_mean) / y_std
z_train_normalized = (z_train - z_mean) / z_std

# Define the model
class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x, y):
        inputs = torch.cat((x, y), dim=1)
        return inputs @ self.W + self.b

    def loss(self, x, y, z):
        return torch.mean(torch.square(self.f(x, y) - z))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.001)

for epoch in range(5000):
    model.loss(x_train_normalized, y_train_normalized, z_train_normalized).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W =", model.W, ", b =", model.b, ", loss =", model.loss(x_train_normalized, y_train_normalized, z_train_normalized))

# Visualization
fig = plt.figure('Linear regression 3D')
ax = fig.add_subplot(111, projection='3d')

x_train_orig = x_train_normalized * x_std + x_mean
y_train_orig = y_train_normalized * y_std + y_mean
z_train_orig = z_train_normalized * z_std + z_mean

ax.scatter(x_train_orig.numpy(), y_train_orig.numpy(), z_train_orig.numpy(), color='blue')

x_grid, y_grid = np.meshgrid(np.linspace(x_train_orig.min().item(), x_train_orig.max().item(), 50),
                             np.linspace(y_train_orig.min().item(), y_train_orig.max().item(), 50))
x_test = torch.from_numpy(x_grid).float().view(-1, 1)
y_test = torch.from_numpy(y_grid).float().view(-1, 1)
z_pred = model.f(x_test, y_test).detach().numpy().reshape(50, 50)

ax.plot_surface(x_grid, y_grid, z_pred, alpha=0.5, color='red')

ax.set_xlabel('Day')
ax.set_ylabel('Length')
ax.set_zlabel('Weight')

# Create manual legend
legend_elements = [Patch(facecolor='blue', label='$(x^{(i)},y^{(i)},z^{(i)})$'),
                   Patch(facecolor='red', label='Regression Plane')]
ax.legend(handles=legend_elements)

plt.show()
