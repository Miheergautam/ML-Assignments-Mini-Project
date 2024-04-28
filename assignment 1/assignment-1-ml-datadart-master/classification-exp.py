import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# Set random seed
np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Q2a: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree
dt = DecisionTreeClassifier()

# Train the Decision Tree on the training set
dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(X_test)

# Q2a: Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Q2a Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Q2b: Nested cross-validation to find optimum depth
depth_range = list(range(1, 11))
kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)

best_depth = None
best_score = 0

for depth in depth_range:
    dt = DecisionTreeClassifier(max_depth=depth)

    # Perform nested cross-validation
    scores = cross_val_score(dt, X, y, cv=kf_inner, scoring='accuracy')

    # Calculate average score
    avg_score = np.mean(scores)

    # Update best depth if the current depth gives a higher average score
    if avg_score > best_score:
        best_score = avg_score
        best_depth = depth

print("Q2b Results:")
print("Best Depth:", best_depth)
print("Best Accuracy:", best_score)