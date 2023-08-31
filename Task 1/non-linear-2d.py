import pandas as pd
import torch
import matplotlib.pyplot as plt



data = pd.read_csv('day_head_circumference.csv', header=0)
print(data)
x_train = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).view(-1, 1)

# Normalize the data.
x_mean = x_train.mean()
x_std = x_train.std()
y_mean = y_train.mean()
y_std = y_train.std()

x_train_normalized = (x_train - x_mean) / x_std
y_train_normalized = (y_train - y_mean) / y_std

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.randn([1, 1], requires_grad=True)
        self.b = torch.randn([1, 1], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))



model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.005)


# Training loop.
for epoch in range(5000):
    model.loss(x_train_normalized, y_train_normalized).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W =", model.W, ", b =", model.b, ", loss =", model.loss(x_train_normalized, y_train_normalized))



# Visualize the results.

# Convert normalized values back to original scale
x_train_orig = x_train_normalized * x_std + x_mean
y_train_orig = y_train_normalized * y_std + y_mean

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')

x_sequence = torch.linspace(torch.min(x_train), torch.max(x_train), steps=100).view(-1, 1)
x_sequence_normalized = (x_sequence - x_mean) / x_std
y_sequence_normalized = model.f(x_sequence_normalized).detach()

# Convert predicted values back to original scale
y_sequence_orig = y_sequence_normalized * y_std + y_mean
x_sequence_orig = x_sequence * x_std + x_mean

plt.plot(x_sequence, y_sequence_orig, label='$f(x) = xW+b$', color='red')
plt.legend()
plt.show()
