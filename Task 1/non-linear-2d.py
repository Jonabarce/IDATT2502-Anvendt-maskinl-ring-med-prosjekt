import pandas as pd
import torch
import matplotlib.pyplot as plt

# Loading data
data = pd.read_csv('day_head_circumference.csv', header=0)

x_train = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).view(-1, 1)

# Normalize the data for x values
x_mean = x_train.mean()
x_std = x_train.std()
x_train_normalized = (x_train - x_mean) / x_std


class NonLinearModel:
    def __init__(self):
        self.W = torch.randn([1, 1], requires_grad=True)
        self.b = torch.randn([1, 1], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid(torch.matmul(x, self.W) + self.b) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = NonLinearModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)  # Adjusted learning rate

# Training loop
for epoch in range(100_000):  # Adjusted epochs
    loss_value = model.loss(x_train_normalized, y_train)
    if epoch % 10_000 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.item()}")

    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad()

print("W =", model.W, ", b =", model.b, ", loss =", model.loss(x_train_normalized, y_train))

# Visualization
plt.figure(figsize=(8, 6))
plt.title("Predict head circumference based on age (in days)")

# Scatter plot of the training data
plt.scatter(x_train, y_train, marker="o")
plt.xlabel("Age (days)")
plt.ylabel("Head Circumference")

# Generate x values and get model predictions
x_sequence = torch.linspace(torch.min(x_train), torch.max(x_train), steps=100).view(-1, 1)
x_sequence_normalized = (x_sequence - x_mean) / x_std
y_sequence = model.f(x_sequence_normalized).detach()

# Plot the predictions
plt.plot(x_sequence, y_sequence, color="orange", linewidth=2, label=r"$f(x) = 20\sigma(xW + b) + 31$")
plt.legend()
plt.show()
