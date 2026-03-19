# Q6 — Convolutional Neural Network with Fashion MNIST

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

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

print("\nTest Accuracy:", test_accuracy)


# ------------- COMMENTS EXPLANATION ---------------
# CNNs are preferred over fully connected networks for image data because they can automatically detect spatial patterns such as edges, shapes, and textures.
# They use fewer parameters and preserve the structure of the image, making them more efficient and effective for image classification tasks.

# The convolutional layer learns filters that detect important features in the images, such as edges or patterns.
# These features help the model distinguish between different clothing categories in the dataset.