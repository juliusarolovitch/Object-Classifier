import torch
import torch.nn as nn
import numpy as np

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

# Define the function to preprocess data
def preprocess_data(data):
    # Normalize the data
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return normalized_data

# Define your training data and labels
training_data = np.array([...])  # Enter 
labels = np.array(['torque_1', 'torque_2', 'angle_1', 'angle_2', 'force_1', 'force_2'])  

# Preprocess the training data
preprocessed_data = preprocess_data(training_data)

# Convert data to PyTorch tensors
inputs = torch.tensor(preprocessed_data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Define the model hyperparameters
input_dim = 6  # Number of data streams
hidden_dim = 64  # Number of LSTM units
output_dim = 3  # Number of object classes

# Create the LSTM model
model = LSTMClassifier(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100  # Adjust as needed
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss for monitoring progress
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Test the model
test_data = np.array([...])  # Replace with your test data
preprocessed_test_data = preprocess_data(test_data)
test_inputs = torch.tensor(preprocessed_test_data, dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    test_outputs = model(test_inputs)
    probabilities = torch.softmax(test_outputs, dim=1)

# Print the probability distribution for each object class
class_names = ["Sphere", "Cylinder", "Cube"]
for i, prob in enumerate(probabilities):
    print(f"Object {i+1}:")
    for j, class_prob in enumerate(prob):
        print(f"{class_names[j]}: {class_prob.item()}")
    print()
