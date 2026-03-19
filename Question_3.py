# Q3 — Controlling Tree Complexity and Interpretability

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split (80:20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree with a constraint (max_depth)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
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

print("TOP 5 IMPORTANT FEATURES:")
print("----------------------------")

# Convert feature importances and names to lists for easier manipulation
importances = list(model.feature_importances_)
names = list(data.feature_names)

# Loop 5 times to find top 5 features
for i in range(5):
    max_value = max(importances)
    index = importances.index(max_value)

    print(names[index], ":", max_value)

    # remove so next max can be found
    importances[index] = -1

# ------------- COMMENTS EXPLANATION ---------------
# Limiting the complexity of the decision tree (for example, by setting a maximum depth) helps prevent overfitting.
# A less complex model is less likely to memorize the training data and can perform better on unseen data.

# Feature importance indicates how much each feature influences the model’s predictions.
# Features with higher importance are used more by the decision tree to make decisions.
# This makes it easier to understand which features are most important, improving the interpretability of the decision tree.