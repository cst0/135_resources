import numpy as np
import time
import matplotlib.pyplot as plt

print(
    """
In the previous example, we saw that logistic regression is an effective method
for binary classification. Now, we'll take a look at a handful of different
gradient descent methods that can be used to optimize the parameters of a
model. Each of these methods has its own strengths and weaknesses, and the
choice of which to use will depend on the specific problem at hand.

This example borrows heavily from the following article, which is another
good resource for learning about this:
https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/ch04.html
"""
)

# just generating some fake data...
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# now we'll add the bias term: the reason we need to do that for this
# example is because we want to use the same code for all three gradient
# descent methods, and the bias term is not automatically added when we
# use the normal equation to solve for the parameters. In real life, you
# would probably just use a library like scikit-learn to calculate the
# parameters for you, and it would take care of the bias term for you.
X_b = np.c_[np.ones((100, 1)), X]


# Function to calculate Mean Squared Error (MSE)
def calculate_mse(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    mse = np.sum((predictions - y) ** 2) / m
    return mse


print(
    """
One common method for optimizing the parameters of a model is called gradient
descent. The idea behind gradient descent is to iteratively update the
parameters in the direction that minimizes the cost function. The cost function
is a measure of how well the model is performing, and the goal is to find the
parameters that minimize this cost function.
"""
)


def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    theta = np.random.randn(n, 1)

    for _ in range(iterations):
        gradients = 2 / m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients

    return theta


print(
    """
Another method for optimizing the parameters of a model is called stochastic
gradient descent. The idea behind stochastic gradient descent is similar to
that of gradient descent, but instead of using the entire dataset to calculate
the gradients at each iteration, we use a single random data point. This can
lead to faster convergence, but it can also be more noisy.
"""
)


def stochastic_gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    theta = np.random.randn(n, 1)

    for _ in range(iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index : random_index + 1]
            yi = y[random_index : random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients

    return theta


print(
    """
A third method for optimizing the parameters of a model is called batch
gradient descent. The idea behind batch gradient descent is to use the entire
dataset to calculate the gradients at each iteration. This can lead to more
stable convergence. However, it can also be slower, especially for large
datasets. We likely won't observe that here though, our dataset is pretty
small.
"""
)


def batch_gradient_descent(X, y, learning_rate, iterations, batch_size):
    m, n = X.shape
    theta = np.random.randn(n, 1)

    for _ in range(iterations):
        for i in range(0, m, batch_size):
            xi = X[i : i + batch_size]
            yi = y[i : i + batch_size]
            gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients

    return theta


print(
    """
With our three gradient descent methods implemented, we can now apply them to
our dataset and compare the results. We'll use the same learning rate and
number of iterations for each method, and we'll also use the same batch size
for the batch gradient descent method. We'll then print the final parameters
for each method and plot the data along with the regression lines that were
learned by each method.

We need to start by defining the learning rate, number of iterations, and batch
size. These are 'hyperparameters' that need to be tuned for each specific
problem, and they can have a big impact on the performance of the model. In
practice, we would typically use techniques like cross-validation to find the
best values for these hyperparameters, but for now we'll just use some
reasonable values.
"""
)
learning_rate = 0.01
iterations = 100
batch_size = 10

# Apply gradient descent methods
theta_gd_start_time = time.time()
theta_gd = gradient_descent(X_b, y, learning_rate, iterations)
theta_gd_end_time = time.time()
print("Time taken for GD: ", theta_gd_end_time - theta_gd_start_time)

theta_sgd_start_time = time.time()
theta_sgd = stochastic_gradient_descent(X_b, y, learning_rate, iterations)
theta_sgd_end_time = time.time()
print("Time taken for SGD: ", theta_sgd_end_time - theta_sgd_start_time)

theta_bgd_start_time = time.time()
theta_bgd = batch_gradient_descent(X_b, y, learning_rate, iterations, batch_size)
theta_bgd_end_time = time.time()
print("Time taken for BGD: ", theta_bgd_end_time - theta_bgd_start_time)

# Print the final parameters
print("True parameters: [4, 3]")
print("Gradient Descent (GD) parameters:", theta_gd.flatten())
print("Stochastic Gradient Descent (SGD) parameters:", theta_sgd.flatten())
print("Batch Gradient Descent (BGD) parameters:", theta_bgd.flatten())

# Plot the data and the regression lines
plt.scatter(X, y, alpha=0.5, label="Data points")
plt.plot(X, X_b.dot(theta_gd), label="GD Regression Line", color="red")
plt.plot(X, X_b.dot(theta_sgd), label="SGD Regression Line", color="green")
plt.plot(X, X_b.dot(theta_bgd), label="BGD Regression Line", color="blue")

# plot the true line
plt.plot(X, 4 + 3 * X, label="True Regression Line", color="black")

plt.title("Comparison of GD, SGD, and BGD")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
