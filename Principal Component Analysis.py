"""
Use the Eigen decomposition available in numpy to do PCA.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Center the data
X_centered = X - np.mean(X, axis=0)

cov_matrix = np.cov(X_centered, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Project data onto the new subspace
k = 3  # number of components
projection_matrix = sorted_eigenvectors[:, :k]
X_transformed = np.dot(X_centered, projection_matrix)

# Plotting the transformed data
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])
plt.cla()

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X_transformed[y == label, 0].mean(),
        X_transformed[y == label, 1].mean() + 1.5,
        X_transformed[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(
    X_transformed[:, 0],
    X_transformed[:, 1],
    X_transformed[:, 2],
    c=y,
    cmap=plt.cm.nipy_spectral,
    edgecolor="k",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.show()
