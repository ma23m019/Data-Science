"""
Create the following data and write to a csv file: Generate 10 random points in each of the the following circles (i) centre at (3,3) and radius 2, (ii) centre at (7,7) and radius 2 (iii) centre at (11,11) and radius 2. Plot the data as well.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Circle
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Function to generate random points in a circle
def generate_points(center, rad, n):
    angles = np.random.uniform(0, 2*np.pi, n)    # Generates n angles between 0 and 2*pi randomly
    distances = np.random.uniform(0, rad, n)     # Generates n distances between 0 and radius randomly
    x = center[0] + distances * np.cos(angles)   # x coordinates of randomly generated points
    y = center[1] + distances * np.sin(angles)   # y coordinates of randomly generated points
    return x, y

# Define circles in the form of dictionaries representing specifications of each circle
circles = [
    {"center": (3, 3), "radius": 2, "n": 10},
    {"center": (7, 7), "radius": 2, "n": 10},
    {"center": (11, 11), "radius": 2, "n": 10}
]

# Save data to CSV
with open('random_points.csv', 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    w.writerow(['X', 'Y'])  # Write header
  
    for c in circles:
        x, y = generate_points(c["center"], c["radius"], c["n"])
        for i in range(len(x)):
            w.writerow([x[i], y[i]])

# Generate data points and plot
plt.figure(figsize=(6, 6))

for c in circles:
    # Plot the circles with dotted boundries
    plt.gca().add_patch(Circle(c["center"], c["radius"], fill=False, color='gray', linestyle='--', linewidth=1))
    
    # Generate points for each circle using the method generate_points
    x, y = generate_points(c["center"], c["radius"], c["n"])
    
    # Plot the generated points
    plt.scatter(x, y, label=f"Center: {c['center']}, Radius: {c['radius']}")
    
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Randomly generated points in in given circles')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

"""
Implement K - means clustering algorithm and for the above data, show the change in the centroid as well as the class assignments. Also, plot the cost function for K varying from 1 to 5. Show that the value of K matches with the intuition from the data. Plot the K-classes for the final K-value.
"""

# Load the data
df1 = pd.read_csv('random_points.csv')
df = pd.DataFrame(df1)
points = df[['X','Y']]
points = np.array(points)

def plot_clusters(points, centroids, assignments, iteration):
    plt.figure(figsize=(4, 3))
    plt.title(f"Iteration {iteration}")
    for i, centroid in enumerate(centroids):
        cluster_points = points[assignments == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
        plt.scatter(centroid[0], centroid[1], marker='x', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def kmeans(points, k, max_iterations=100):
    n = points.shape[0]
    # Initialize centroids randomly
    centroids = points[np.random.choice(n, k, replace=False)]
    assignments = np.zeros(n, dtype=int)
    
    for iteration in range(max_iterations):
        # Assign points to nearest centroid
        for i, point in enumerate(points):
            distances = np.linalg.norm(centroids - point, axis=1)
            assignments[i] = np.argmin(distances)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = points[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
        # Plot clusters at each iteration
        plot_clusters(points, centroids, assignments, iteration + 1)
    
    return centroids, assignments

# Perform K-means clustering for k = 1, 2, 3, 4, 5
for k in range(1, 6):
    print(f"\n\nPerforming K-means clustering for k = {k}")
    centroids, assignments = kmeans(points, k)
    print(f"\nFinal centroids for k = {k}:")
    print(centroids)
    print(f"\nClass assignments for k = {k}:")
    print(assignments)

# ============================= GRAPH OF COST FUNCTION =============================

# Modify the kmeans function to return centroids, assignments, cost
def kmeans(points, k, max_iterations=100):
    n = points.shape[0]
    # initialize centroids randomly
    centroids = points[np.random.choice(n, k, replace=False)]
    assignments = np.zeros(n, dtype=int)
    
    for iteration in range(max_iterations):
        # assign points to nearest centroid
        for i, point in enumerate(points):
            distances = np.linalg.norm(centroids - point, axis=1)
            assignments[i] = np.argmin(distances)
        
        # update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = points[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # compute cost (sum of squared distances)
    cost = 0
    for i, point in enumerate(points):
        cost += np.linalg.norm(point - centroids[assignments[i]]) ** 2
    
    return centroids, assignments, cost

# Compute cost function for K varying from 1 to 5
costs = []
for k in range(1, 6):
    centroids, assignments, cost = kmeans(points, k)
    costs.append(cost)

# Plot the cost function
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), costs, marker='o')
plt.title('Cost function vs. Number of clusters (K)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Cost function')
plt.grid(True)
plt.show()

# ================================ FINAL PLOT FOR K = 3 ================================
    
def plot_clusters(points, centroids, assignments):
    plt.figure(figsize=(8, 6))
    plt.title('Final Clusters for K = 3')
    for i, centroid in enumerate(centroids):
        cluster_points = points[assignments == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
        plt.scatter(centroid[0], centroid[1], marker='x', color='red', label=f'Centroid {i + 1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# compute centroids and assignments for K=3
centroids, assignments = kmeans(points, 3)

# plot final clusters
plot_clusters(points, centroids, assignments)

"""
Taking any two classes from the above data, add labels to them (0 or 1) and create a new csv file. Split the data into Train / Test set as 70/30. (a) Plot the decision boundary using logistic regression. (b) Evaluate the metrics such as Precision, Recall, F1-Score and Accuracy on the test data without using any library.
"""

# ----------------------- (a) -----------------------
# Assign labels
df_classes = df.assign(Label=[0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2])

# Select data with labels 0 or 1
two_classes = df_classes['Label'].isin([0,1])
df_two_classes = df_classes[two_classes]

# Extract features and target variable
X = df_two_classes[['X', 'Y']]
y = df_two_classes['Label']

# Add intercept term
X['intercept'] = 1

# Split the data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define gradient function
def compute_gradient(X, y, w):
    m = len(y)
    h = sigmoid(np.dot(X, w))  # Hypothesis function
    gradient = (1/m) * np.dot(X.T, h - y)
    return gradient

# Gradient descent to compute the weights
def gradient_descent(X, y, w, learning_rate, num_iterations):    
    for _ in range(num_iterations):
        gradient = compute_gradient(X, y, w)        
        w -= learning_rate * gradient        
    return w

# Define cost function
def cost_function(w, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, w))
    J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return J

# Initialize weights
w = np.zeros(X_train.shape[1])

learning_rate = 0.01   # Set constant learning rate
num_iterations = 6500  # Set number of iterations

# Optimize weights using gradient descent
w = gradient_descent(X_train, y_train, w, learning_rate, num_iterations)

# Plot decision boundary
x_values = np.array([np.min(X_train.iloc[:, 1]), np.max(X_train.iloc[:, 1])])
y_values = - (w[0] * x_values + w[2]) / w[1]

plt.figure(figsize=(6,6))
plt.scatter(X_train['X'], X_train['Y'], label='Training Data')
plt.scatter(X_test['X'], X_test['Y'], marker='x', label='Test Data')
plt.plot(x_values, y_values, color='red', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary using Logistic Regression')
plt.legend()

plt.show()

# ----------------------- (b) -----------------------
# Define a function to make predictions
def predict(X, w):
    z = np.dot(X, w)
    return sigmoid(z)

# Make predictions on the test set
y_pred_prob = predict(X_test.values, w)
y_pred = np.round(y_pred_prob)

# Calculate True Positives, False Positives, True Negatives, False Negatives
TP = np.sum((y_pred == 1) & (y_test == 1))
FP = np.sum((y_pred == 1) & (y_test == 0))
TN = np.sum((y_pred == 0) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

# Calculate the metrics
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Print the metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Accuracy:", accuracy)
