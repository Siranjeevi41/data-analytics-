"""
* We generate synthetic data representing student marks for multiple students over several semesters.
* We use an LSTM layer followed by a Dense layer to build the RNN model.
* We compile the model with mean squared error (MSE) loss and train it on the input-output pairs.
* Finally, we use the trained model to predict the marks of the next semester for a given student.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate synthetic data for demonstration
# Let's say we have data of student marks for 10 students over 5 semesters
num_students = 10
num_semesters = 5
num_subjects = 3

# Generate random student marks between 0 and 100 for each subject
student_marks = np.random.randint(0, 100, size=(num_students, num_semesters, num_subjects)).astype(np.float32)

# Normalize the marks
max_marks = 100.0
student_marks /= max_marks

# Let's say our task is to predict the marks of the next semester based on previous semesters
# For simplicity, we'll predict the marks of the last semester based on the previous ones
X = student_marks[:, :-1, :]  # Input: marks of first 4 semesters
y = student_marks[:, -1, :]   # Output: marks of the last semester

# Define the RNN model
model = Sequential([
    LSTM(64, input_shape=(num_semesters-1, num_subjects)),
    Dense(num_subjects)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=50, batch_size=2)

# After training, you can use the model to predict the marks of the next semester for new students or existing students
# For example, let's predict marks for the first student's next semester
student = X[0:1]  # Data of the first student
predicted_marks = model.predict(student)
predicted_marks *= max_marks  # Scale back to original range

print("Predicted marks for next semester:")
print(predicted_marks)
