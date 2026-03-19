# Q1 — Dataset Exploration and Understanding

from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Construct feature matrix X and target vector y
X = data.data
y = data.target

# Report the shape of X and y
print("\nFeature matrix (X) shape:", X.shape)
print("Target vector (y) shape:", y.shape)

# Report the number of samples belonging to each class
print("\n-------------------------------")
print("Number of samples in each class")
print("-------------------------------")
print("No. of malignant samples:", sum(y == 0))
print("No. of benign samples:", sum(y == 1))


# ------------- COMMENTS EXPLANATION ---------------

# The dataset contains 569 samples and 30 features.
# The target variable has two classes:
# 0 = malignant (cancerous)
# 1 = benign (non-cancerous)

# The dataset is slightly imbalanced, as there are more benign samples than malignant samples.

# Class balance is important because if one class has significantly more samples, the model may become biased toward that class. This can result in high accuracy
# but poor performance on the minority class. In medical applications, correctly identifying malignant cases is important, as misclassification can have serious consequences.