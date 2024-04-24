"""
Implement the forward propagation for a two hidden layer network. Initialize the weights randomly. You can choose the number  of neurons in the hidden layer and use sigmoid activation function. Report the evaluation metrics for the network.  Also use other non-linear activation functions like ReLU and Tanh. Report the loss using both MSE and Cross Entropy.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Initialize wights and biases randomly
def initialize_weights(input_size, layer1_size, layer2_size, output_size):
    np.random.seed(9)
    W1 = np.random.randn(layer1_size, input_size)
    W2 = np.random.randn(layer2_size, layer1_size)
    W3 = np.random.randn(output_size, layer2_size)
    
    b1 = np.zeros((layer1_size, 1))
    b2 = np.zeros((layer2_size, 1))
    b3 = np.zeros((output_size, 1))
    
    return W1, b1, W2, b2, W3, b3

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2, W3, b3, activation):
    # Hidden layer 1
    Z1 = np.dot(W1, X.T) + b1
    A1 = activation(Z1)
    
    # Hidden layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = activation(Z2)
    
    # Output layer
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    return A3

# Loss function for computing MSE loss and cross entropy loss
def loss_function(y_pred_prob):
    y_pred = np.round(y_pred_prob)    # Roud the y probabilities to 0 or 1
    
    mse_loss = np.mean((y_pred - y)**2)
    cross_entropy_loss = - np.mean(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))
    
    print(f"MSE Loss = {mse_loss}")
    print(f"Cross Entropy Loss = {cross_entropy_loss}")

# Evaluation metrics
def evaluation_metric(y_pred_prob):
    y_pred = np.round(y_pred_prob)     # Roud the y probabilities to 0 or 1

    # Calculate True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((y_pred == 1) & (y == 1))
    FP = np.sum((y_pred == 1) & (y == 0))
    TN = np.sum((y_pred == 0) & (y == 0))
    FN = np.sum((y_pred == 0) & (y == 1))

    # Calculate the metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1_score}")
    print(f"Accuracy = {accuracy}")   

# Load the dataset
df = pd.read_csv('Logistic_regression_ls.csv')
X = df[['x1', 'x2']].values
y = df['label'].values

# Initialize the parameters
input_size = X.shape[1]
layer1_size = 150
layer2_size = 70
output_size = 1

# Normalization the data using StandardScaler
standard_scaler = StandardScaler()
X_normalized_standard = standard_scaler.fit_transform(X)

# Initialize the weights and biases
W1, b1, W2, b2, W3, b3 = initialize_weights(input_size, layer1_size, layer2_size, output_size)

# =========================== SIGMOID ACTIVATION ==========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("SIGMOID ACTIVATION")

# Forward propagation for sigmoid activation
pred_sigmoid_prob = forward_propagation(X_normalized_standard, W1, b1, W2, b2, W3, b3, sigmoid)
# Loss functions for sigmoid activation
loss_function(pred_sigmoid_prob)
# Evaluation metrics for sigmoid activation
evaluation_metric(pred_sigmoid_prob)

# =========================== Tanh ACTIVATION =========================
def tanh(z):
    return np.tanh(z)

print("\nTanh ACTIVATION")

# Forward propagation for tanh activation
pred_tanh_prob = forward_propagation(X_normalized_standard, W1, b1, W2, b2, W3, b3, tanh)
# Loss functions for tanh activation
loss_function(pred_tanh_prob)
# Evaluation metrics for tanh activation
evaluation_metric(pred_tanh_prob)

# =========================== ReLU ACTIVATION ===========================
def ReLU(z):
    return np.maximum(0,z)

print("\nReLU ACTIVATION")

# Forward propagation for ReLU activation
pred_ReLU_prob = forward_propagation(X_normalized_standard, W1, b1, W2, b2, W3, b3, ReLU)
# Loss functions for ReLU activation
loss_function(pred_ReLU_prob)
# Evaluation metrics for ReLU activation
evaluation_metric(pred_ReLU_prob)
