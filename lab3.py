import torch
import torch.nn as nn
import torch.optim as optim


class FeedforwardNN(nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim):
    super(FeedforwardNN, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out


input_dim = 10
hidden_dim = 32
output_dim = 1

model = FeedforwardNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x_dummy = torch.randn(64, input_dim)
y_dummy = torch.randn(64, output_dim)

optimizer.zero_grad()
outputs = model(x_dummy)
loss = criterion(outputs, y_dummy)
loss.backward()
optimizer.step()