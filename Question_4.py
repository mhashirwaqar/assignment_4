# Q4 — Neural Network for Binary Classification

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split (80:20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network (1 hidden layer + sigmoid output)
model = models.Sequential()

# Hidden layer
model.add(layers.Dense(10, input_shape=(X_train.shape[1],)))

# Output layer
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, verbose=1)

# Predictions
y_train_pred = (model.predict(X_train) > 0.5).astype(int)
y_test_pred = (model.predict(X_test) > 0.5).astype(int)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Report results
print("\n----------------------------")
print("TRAINING ACCURACY:", train_accuracy)
print("TESTING ACCURACY:", test_accuracy)
print("----------------------------")


# ------------- COMMENTS EXPLANATION ---------------
# Feature scaling is necessary for neural networks because they are sensitive to the scale of input features.
# If features are on very different scales, some features may dominate others, making learning inefficient or unstable.
# Standardizing the data ensures that all features contribute equally and helps the model converge faster and more reliably.

# An epoch represents one complete pass through the entire training dataset during the training process.
# During each epoch, the model processes all training examples and updates its weights to reduce error.
# Multiple epochs allow the model to gradually improve its performance by learning patterns in the data over time.