# Q5 — Model Evaluation and Comparison

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split (same as previous questions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree (from Q3)
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

y_test_pred_dt = dt_model.predict(X_test)

# Neural Network (from Q4)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Neural Network
nn_model = models.Sequential()
nn_model.add(layers.Dense(10, input_shape=(X_train_scaled.shape[1],)))
nn_model.add(layers.Dense(1, activation='sigmoid'))

# Compile
nn_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train
nn_model.fit(X_train_scaled, y_train, epochs=20, verbose=0)

# Predictions
y_test_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype(int)

# Confusion Matrices
print("\n-----------------------------")
print("CONFUSION MATRIX")
print("-----------------------------")
print("Decision Tree:")
print(confusion_matrix(y_test, y_test_pred_dt))
print("-----------------------------")
print("Neural Network:")
print(confusion_matrix(y_test, y_test_pred_nn))


# ------------- COMMENTS EXPLANATION ---------------
# The confusion matrix shows how many predictions were correct and incorrect for each class.
# It provides more detailed insight than accuracy by showing false positives and false negatives.

# The neural network is generally preferred for this task because it can learn more complex patterns and often achieves better predictive performance.

# An advantage of the decision tree is that it is easy to understand and interpret.
# A limitation is that it can overfit the data if not properly constrained.

# An advantage of the neural network is its ability to model complex relationships in the data.
# A limitation is that it is less interpretable and behaves like a black box.