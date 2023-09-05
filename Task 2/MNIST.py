import matplotlib.pyplot as plt
import torch
import torchvision


# Load MNIST data
mnist_train = torchvision.datasets.MNIST("./data", train=True, download=True)
x_train = (mnist_train.data.float().reshape(-1, 784) / 255 - 0.5)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.MNIST("./data", train=False, download=True)
x_test = (mnist_test.data.float().reshape(-1, 784) / 255 - 0.5)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

class MNISTModel:
    def __init__(self):
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    def logits(self, x):
        return torch.matmul(x, self.W) + self.b

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

def visualize_results(model, x_test, y_test):
    plt.figure(figsize=(10, 5))  # Set the figure size

    for i in range(10):
        index = torch.randint(0, x_test.shape[0], ())
        x = x_test[index]

        plt.subplot(2, 5, i + 1)  # Create a subplot in a 2x5 grid
        plt.imshow(x.reshape(28, 28), cmap="gray")
        plt.title(f"Model: {model.f(x).argmax().item()}\nActual: {y_test[index].argmax().item()}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


model = MNISTModel()

# Training loop
optimizer = torch.optim.SGD([model.W, model.b], lr=0.1)
num_epochs = 2000

for epoch in range(num_epochs):
    # Forward pass
    loss = model.loss(x_train, y_train)
    print(f"Epoch {epoch + 1}: Loss = {loss:.3f}, Accuracy = {model.accuracy(x_train, y_train):.3f}")
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Final Accuracy: {model.accuracy(x_train, y_train).item()}")
visualize_results(model, x_test, y_test)










