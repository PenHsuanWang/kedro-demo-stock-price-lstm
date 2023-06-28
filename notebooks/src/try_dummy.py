#%%

import matplotlib.pyplot as plt

# Dummy time series data
training_data = [32.1, 35.2, 34.7, 36.5, 38.9, 40.2, 42.1, 39.8, 37.2, 34.6, 
                31.5, 29.4, 27.6, 30.2, 32.8, 35.7, 37.9, 40.4, 42.3, 45.1, 
                43.2, 40.8, 38.5, 36.1, 34.2, 31.9, 29.7, 27.3, 30.8, 33.5, 
                36.2, 38.1, 41.2, 39.4, 36.7, 34.3, 31.7, 29.5, 27.8, 31.4, 
                34.8, 37.6, 40.1, 42.5, 44.3, 41.9, 39.2, 36.8, 34.1, 31.6, 
                29.3, 27.7, 30.6, 33.3, 36.4, 38.7, 40.9, 43.5, 45.6, 43.7, 
                41.3, 38.4, 35.9, 33.8, 31.3, 29.1, 27.5, 30.4, 33.2, 36.3, 
                38.8, 41.6, 44.2, 42.9, 40.7, 37.8, 35.3, 33.1, 30.9, 28.7, 
                27.2, 29.8, 32.6, 35.1, 37.3, 40.3, 43.1, 45.3, 42.6, 39.9, 
                37.4, 35.6, 33.4, 31.2, 29.6, 27.9, 29.2, 32.7, 35.5, 37.7, 
                40.6, 43.4, 44.9, 42.4, 39.7, 37.1, 34.5, 32.9, 30.7, 28.8]

# Plotting the time series data
plt.plot(training_data)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Data')
plt.show()

# %%

import torch
import torch.nn as nn

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Prepare the input data
input_size = 1  # Number of features (in this case, it's univariate)
output_size = 1  # Number of output predictions
hidden_size = 32  # Number of LSTM units
num_layers = 2  # Number of LSTM layers

# Convert the training data to tensors
training_data = torch.tensor(training_data).view(-1, 1).float()

# Prepare the input and target data for training
input_sequence = training_data[:-1]
target_sequence = training_data[1:]

# Reshape input and target sequences to (batch_size, sequence_length, input_size)
input_sequence = input_sequence.view(input_sequence.size(0), 1, input_size)
target_sequence = target_sequence.view(target_sequence.size(0), output_size)

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the LSTM model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    outputs = model(input_sequence)
    print(input_sequence)
    loss = criterion(outputs, target_sequence)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Set the model in evaluation mode
model.eval()

# Generate predictions for the entire training data
predicted_sequence = model(input_sequence)

# Plot the original data and the predicted values
plt.plot(training_data.numpy(), label='Original Data')
plt.plot(predicted_sequence.detach().numpy(), label='Predicted Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Data vs Predicted Data')
plt.legend()
plt.show()

# %%

predicted_sequence
# %%
