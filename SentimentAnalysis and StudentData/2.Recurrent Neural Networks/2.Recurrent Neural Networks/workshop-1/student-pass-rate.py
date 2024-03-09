"""
To predict student pass rates based on historical data

*   We load historical student data from a CSV file, focusing on relevant features such as exam scores, attendance, and study hours.
*   We normalize the data using Min-Max scaling.
*   We create input-output sequences where each input sequence contains a fixed number of past time steps (sequence_length) for the selected features, and the corresponding output is the pass rate at the next time step.
*   We split the data into training and testing sets.
*   We define an LSTM-based RNN model with one LSTM layer followed by a Dense layer for regression.*
*   We compile and train the model on the training data, monitoring the training and validation loss.
*   We evaluate the model's performance on the test data and make predictions.
*   Finally, we plot the ground truth pass rates against the predicted pass rates to visualize the model's performance.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load student data (e.g., exam scores, attendance, study hours, etc.)
# Assume the data is stored in a CSV file with columns: date, exam_score, attendance, study_hours, pass_rate, etc.
data = pd.read_csv("student_data.csv")

# Select the relevant features for prediction
features = ['exam_score', 'attendance', 'study_hours']  # Example features
target = 'pass_rate'

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + [target]])

# Split the data into input (X) and output (y) sequences
sequence_length = 10  # Number of past time steps to consider
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length][-1])  # Target is the last column (pass_rate)

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, len(features))),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Make predictions on the test data
predictions = model.predict(X_test)

# Inverse transform the predictions and ground truth to the original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the predictions vs. ground truth
plt.plot(y_test, label='Ground Truth')
plt.plot(predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Pass Rate')
plt.legend()
plt.show()

