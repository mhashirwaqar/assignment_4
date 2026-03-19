# Q2 — Decision Tree Model Using Entropy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split (80:20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree with entropy criterion
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Report results
print("\n----------------------------")
print("TRAINING ACCURACY:", train_accuracy)
print("TESTING ACCURACY:", test_accuracy)
print("----------------------------")

# ------------- COMMENTS EXPLANATION ---------------

# Entropy measures how mixed or impure the data is. In a decision tree, it is used to decide the best way to split the data. The model chooses splits that reduce entropy, making the groups more pure.

# If the training accuracy is much higher than the test accuracy, it means the model is overfitting. This means it memorizes the training data but does not perform well on new data.

# If both training and test accuracy are close and high, it means the model generalizes well and performs consistently on both training and unseen data.