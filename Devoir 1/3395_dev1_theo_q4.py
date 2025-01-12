# -*- coding: utf-8 -*-
"""3395-dev1-theo-q4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1j2w5mQEu8DZ26ptNP91YJWJqXCUex4Sa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the 6 data points (2D) and their labels
X = np.array([[5, -2], [2, -10], [2, -15], [8, 0], [2, 5], [8, 5]])
y = np.array([1, 1, 1, 2, 2, 2])  # 1 for one class, 2 for another

# Create and train the 1-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Create a mesh to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict classifications for each point in the mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and points
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title('k plus proches')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

# Calculate the mean of each class
class_1_mean = X[y == 1].mean(axis=0)
class_2_mean = X[y == 2].mean(axis=0)

# Create a meshgrid for the plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Calculate the distance of each point in the meshgrid to the class means
dist_to_class_0 = np.sqrt((xx - class_1_mean[0])**2 + (yy - class_1_mean[1])**2)
dist_to_class_1 = np.sqrt((xx - class_2_mean[0])**2 + (yy - class_2_mean[1])**2)

# Decision boundary is where the distances to the two class means are equal
decision_boundary = dist_to_class_0 - dist_to_class_1

# Plot the datapoints
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.RdYlBu, edgecolors='k')

# Plot the centroids (means) of each class
plt.scatter(class_1_mean[0], class_1_mean[1], s=200, c='red', marker='X')
plt.scatter(class_2_mean[0], class_2_mean[1], s=200, c='blue', marker='X')

# Plot the decision boundary (the line where dist_to_class_0 == dist_to_class_1)
plt.contour(xx, yy, decision_boundary, levels=[0], linewidths=2, colors='black')

# Add titles and labels
plt.title("Decision Line Based on Class Means")
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()

# Show plot
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the 6 data points (2D) and their labels
X = np.array([[5, -2], [2, -10], [2, -15], [8, 0], [2, 5], [8, 5], [3,1], [-3, 0], [5, -8]])
y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])  # 1 for one class, 2 for another, 3 for another

# Create and train the 1-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Create a mesh to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict classifications for each point in the mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and points
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title('k plus proches')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

# Calculate the mean of each class
class_1_mean = X[y == 1].mean(axis=0)
class_2_mean = X[y == 2].mean(axis=0)
class_3_mean = X[y == 3].mean(axis=0)  # Added for third class

# Create a meshgrid for the plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Calculate the distance of each point in the meshgrid to the class means
dist_to_class_1 = np.sqrt((xx - class_1_mean[0])**2 + (yy - class_1_mean[1])**2)
dist_to_class_2 = np.sqrt((xx - class_2_mean[0])**2 + (yy - class_2_mean[1])**2)
dist_to_class_3 = np.sqrt((xx - class_3_mean[0])**2 + (yy - class_3_mean[1])**2)  # Added for third class

# Assign class labels based on the shortest distance to the mean
Z = np.argmin(np.array([dist_to_class_1, dist_to_class_2, dist_to_class_3]), axis=0)

# Plot the decision boundary and datapoints
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolor='k', cmap=plt.cm.RdYlBu)

# Plot the datapoints
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.RdYlBu, edgecolors='k')

# Plot the centroids (means) of each class
plt.scatter(class_1_mean[0], class_1_mean[1], s=200, c='red', marker='X')
plt.scatter(class_2_mean[0], class_2_mean[1], s=200, c='yellow', marker='X')
plt.scatter(class_3_mean[0], class_3_mean[1], s=200, c='blue', marker='X')  # Added for third class

# Add titles and labels
plt.title("Decision Boundaries Based on Class Means")
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()

# Show plot
plt.show()