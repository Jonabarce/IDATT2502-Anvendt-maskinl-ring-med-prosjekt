import torch
import torch.nn as nn
import numpy as np

# New data format
emojis = {
    'hat': '🎩',
    'rat': '🐭',
    'cat': '😺',
    'flat': '🏢',
    'matt': '🙋',
    'cap': '🧢',
    'son': '👦'
}

index_to_emoji = [value for key, value in emojis.items()]

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

# Character encodings using np.eye
char_encodings = np.eye(len(index_to_char))

encoding_size = len(char_encodings)

# Emoji encodings using np.eye
emoji_encodings = np.eye(len(index_to_emoji))

x_train_numpy = np.array([
    [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # Example 1
    [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # Example 2
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],  # Example 3
    [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],  # Example 4
    [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],  # Example 5
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],  # Example 6
    [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]],  # Example 7
])

y_train_numpy = np.array([
    [emoji_encodings[0]],  # Target for Example 1
    [emoji_encodings[1]],  # Target for Example 2
    [emoji_encodings[2]],  # Target for Example 3
    [emoji_encodings[3]],  # Target for Example 4
    [emoji_encodings[4]],  # Target for Example 5
    [emoji_encodings[5]],  # Target for Example 6
    [emoji_encodings[6]],  # Target for Example 7
])

# Convert training data to torch tensors
x_train = torch.tensor(x_train_numpy, dtype=torch.float32)
y_train = torch.tensor(y_train_numpy, dtype=torch.float32)

# Define the LSTM model class
class ManyToOneLSTMModel(nn.Module):
    def __init__(self, encoding_size):
        super(ManyToOneLSTMModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, len(index_to_emoji))  # predicting the emojis

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), torch.argmax(y, dim=1))

# Data

# Training
model = ManyToOneLSTMModel(encoding_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    for x, y in zip(x_train, y_train):
        model.reset()
        optimizer.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Testing
def test(word):
    model.reset()  # Reset model
    tensor_list = []
    for char in word:  # Loop through text
        idx = index_to_char.index(char)  # Get character index
        tensor_list.append([char_encodings[idx]])  # Append to list

    # Convert list to numpy array and then to tensor
    inp_numpy = np.array(tensor_list)
    inp = torch.tensor(inp_numpy, dtype=torch.float)

    out = model.f(inp)  # Get prediction
    last_out = out[-1]  # Take the last output in the sequence
    return index_to_emoji[last_out.argmax().item()]



print(test("rat") + " should be 🐭")
print(test("cap") + " should be 🧢")
print(test("matt") + " should be 🙋")
print(test("son") + " should be 👦")
print(test("hat") + " should be 🎩")
print(test("cat") + " should be 😺")
print(test("flat") + " should be 🏢")


