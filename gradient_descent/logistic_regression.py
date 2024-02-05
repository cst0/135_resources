# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix

print(
    """
Let's make some predictions using a logistic regression model. We'll start by
creating a synthetic dataset and then train a logistic regression model on the
training data. Finally, we'll evaluate the model's performance on the testing
data and visualize the decision boundary.

Conveniently, scikit-learn provides a make_classification function to generate
synthetic datasets for classification problems. We'll use this function to
create a dataset with 2 informative features and no redundant features, and
we'll explore some real data in a different example.
"""
)

X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
)

print(
    """
We'll need to split the dataset into training and testing sets, so we can
train the model on the training data and evaluate its performance on data
the model has never seen before. We'll use 80% of the data for training and
20% for testing.
"""
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # test_size of .2 is 20 test, 80 train.
    random_state=42,  # random_state is for reproducibility
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print(
    """
scikit-learn provides a LogisticRegression class to create a logistic
regression mode, so we don't have to code it from scratch. Logistic regression
is a suitable model for binary classification problems, where the target
variable has two possible classes. But it can also be used for multi-class
classification problems.
"""
)
model = LogisticRegression()

print(
    """
With a model instance created, we can now train the model on the training data
using the 'fit' method. This method adjusts the model's parameters to minimize
the error on the training data. Once the model is trained, we can use it to
make predictions on the testing data.

Notice that the prediction is being done on data that the model has never
seen before! This helps show how well the model generalizes to new data.

The model's performance can be evaluated using metrics such as accuracy and
confusion matrix. The accuracy is the proportion of correctly classified
samples, and the confusion matrix shows the number of true and false
positives and negatives.
"""
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

print(
    """
The accuracy and confusion scores on novel data are good, but we can also
visualize the decision boundary to better understand the model's behavior. The
decision boundary is the line that separates the two classes, and it's where
the model's output is exactly 0.5. We can plot the decision boundary along with
the training data to see how the model makes predictions.
"""
)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, marker="o", edgecolors="k")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
