# Q7 — CNN Error Analysis and Misclassification Study

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Question 6 Trained CNN:

# Load dataset
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize pixel values to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to include channel dimension (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = models.Sequential()

# Convolutional layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))

# MaxPooling layer
model.add(layers.MaxPooling2D((2,2)))

# Flatten before Dense layer
model.add(layers.Flatten())

# Output layer (10 classes)
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model (at least 15 epochs)
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Question 7 Error Analysis:

# Generate predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Find misclassified images
misclassified = np.where(y_pred != y_test)[0]

print("\nDisplaying 3 Misclassified Images:")

# Show 3 misclassified images
for i in range(3):
    index = misclassified[i]

    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title("True: " + class_names[y_test[index]] +
              " | Pred: " + class_names[y_pred[index]])
    plt.axis('off')
    plt.show()


# ------------- COMMENTS EXPLANATION ---------------
# One pattern observed in the misclassifications is that the model often confuses
# visually similar items such as shirts, pullovers, and coats, since they share similar shapes and textures.

# One realistic way to improve the CNN performance is to increase the model complexity by adding more convolutional
# layers or filters, allowing the model to learn more detailed features and better distinguish between similar classes.
