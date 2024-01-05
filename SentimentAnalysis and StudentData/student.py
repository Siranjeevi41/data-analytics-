"""
* We load the dataset and split it into features (X) and the target variable (y).
* The features are standardized using StandardScaler.
* An SVM model with a linear kernel is trained on the training set.
* Model performance is evaluated using accuracy, confusion matrix, and classification report.
* The decision boundary and support vectors are plotted to visualize how the SVM separates the pass and fail instances.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("C:/Users/siranjeevi/Dropbox/PC/Downloads/code/code/student_scores.csv")

# Extract features (exam scores) and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Create an SVM model with a linear kernel
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_std, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_std)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Plot support vectors
ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('SVM Decision Boundary for Student Pass/Fail Prediction')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()
