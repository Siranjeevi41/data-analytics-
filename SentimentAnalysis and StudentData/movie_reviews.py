#(The goal is to build an SVM model to classify these reviews based on their textual content,
# We load a dataset of movie reviews and split it into training and testing sets.
# Text data is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
# An SVM model with a linear kernel is trained on the TF-IDF transformed training set.
# Model performance is evaluated using accuracy, confusion matrix, and classification report.
# We make predictions on sample movie reviews to demonstrate the model's sentiment classification)
# Import necessary libraries
# Import necessary libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the movie reviews dataset
data = pd.read_csv("C:/Users/siranjeevi/Dropbox/PC/Downloads/code/code/movie_reviews.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create an SVM model
svm_model = LinearSVC(C=1)
svm_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Investigate Misclassifications
misclassified_indices = np.where(y_test != y_pred)[0]
print("\nMisclassified Samples:")
for idx in misclassified_indices:
    print(f"Review: {X_test.iloc[idx]}\nActual Sentiment: {y_test.iloc[idx]}\nPredicted Sentiment: {y_pred[idx]}\n")

# Explore TF-IDF Features for Misclassified Samples
print("\nTF-IDF Features for Misclassified Samples:")
for idx in misclassified_indices:
    print(f"Review: {X_test.iloc[idx]}\nTF-IDF Vector: {X_test_tfidf[idx]}\nActual Sentiment: {y_test.iloc[idx]}\nPredicted Sentiment: {y_pred[idx]}\n")

# Fine-Tune Model Parameters (example: try different values for C)
# svm_model = LinearSVC(C=0.1)  # Adjust C value as needed
# svm_model.fit(X_train_tfidf, y_train)
# y_pred_tuned = svm_model.predict(X_test_tfidf)
# print("\nFine-Tuned Model:")
# print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
# print("Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Example Predictions
sample_reviews = ["I loved the movie, it was fantastic!", "The plot was confusing and the acting was terrible."]
sample_reviews_tfidf = vectorizer.transform(sample_reviews)
sample_predictions = svm_model.predict(sample_reviews_tfidf)

print("\nSample Predictions:")
for i, review in enumerate(sample_reviews):
    print(f"Review: {review}\nPredicted Sentiment: {'Negative' if sample_predictions[i] == 1 else 'Positive'}\n")
