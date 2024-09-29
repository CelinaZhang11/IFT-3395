import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the 6 data points (2D) and their labels
X = np.array([[5, -2], [2, -10], [2, -15], [8, 0], [2, 5], [8, 5]])
y = np.array([1, 1, 1, 2, 2, 2])  # 0 for one class, 1 for another

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
plt.title('k plus proches voisins')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()