import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

print(
    """
Let's try multi-class classification using logistic regression on some
real data. We'll use the Iris dataset, which is a classic dataset in
machine learning. It contains 150 samples of iris flowers, each with
four features: sepal length, sepal width, petal length, and petal width.
The goal is to predict the species of the flower based on these features.
"""
)

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Select the first two features for visualization
X_visualize = X[:, :2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_visualize, y, test_size=0.2, random_state=42
)

print(
    f"""
In our dataset, we see that there are {X.shape[0]} samples and {X.shape[1]}
features. We're only going to use the first two features for visualization
purposes (so we can keep it 2d), but we'll use all four features for training
and testing our model.

It's convenient to scale the features to have a mean of 0 and a standard
deviation of 1. This can help the model converge more quickly and can also help
with the interpretation of the model's coefficients. We'll use the
StandardScaler from scikit-learn to do this, and then we'll split the data into
training and testing sets before passing it to the logistic regression model.
"""
)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(
    """
Now that we've scaled the features, we can train the logistic regression model.
We're using the 'multinomial' option for the multi_class parameter, because we
have more than two classes in our target variable. We're also setting the
max_iter parameter to 1000 to ensure that the model converges, and we're
setting the random_state parameter to 42 to ensure reproducibility.
"""
)
model = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

print(
    """
Like the other logistic regression model we trained, this model also outputs
the probabilities of each class for each sample. We can use these probabilities
to visualize the decision regions of the model, which can help us understand
how the model is making predictions. We'll plot the decision regions for the
first two features of the Iris dataset, and we'll also plot the training points
to see how the model is classifying them.
"""
)
h = 0.02  # step size in the mesh
x_min, x_max = X_visualize[:, 0].min() - 1, X_visualize[:, 0].max() + 1
y_min, y_max = X_visualize[:, 1].min() - 1, X_visualize[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the training points
plt.scatter(
    X_visualize[:, 0], X_visualize[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired
)

plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.title("Logistic Regression Decision Regions")
plt.show()
