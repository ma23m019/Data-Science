"""
1. Implement SVM to classify the type of iris flower based on its sepal length and width using the iris dataset.
2. Also try to use the scikit-learn digits dataset and an SVM to classify handwritten digits.

For both datasets, provide a step-by-step code, including:  

1. Loading the dataset 
2. Visualizing the data 
3. Splitting the data into training and testing sets 
4. Initializing and training the SVM model 
5. Testing the model 
"""
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

# ==================================================== IRIS DATASET ====================================================
# ---------------------------------- STEP 1: Loading the dataset ----------------------------------
from sklearn.datasets import load_iris

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

# ---------------------------------- STEP 2: Visualizing the data ----------------------------------
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue=iris.target_names[target], data=X)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Sepal Features')
plt.show()

# ------------------------------- STEP 3: Splitting the data into training and testing sets -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X[['sepal length (cm)', 'sepal width (cm)']], target, test_size=0.2, random_state=42)

# ---------------------------------- STEP 4: Initializing and training the SVM model ----------------------------------
# Initialize the SVM model
svm_model = SVC(kernel='linear', C=1)

# Train the SVM model on the training data
svm_model.fit(X_train, y_train)

# ---------------------------------- STEP 5: Testing the model ----------------------------------
# Predict the labels for the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# ============================================== HANDWRITTEN DIGITS DATASET ==============================================
# ---------------------------------- STEP 1: Loading the dataset ----------------------------------

from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

# ---------------------------------- STEP 2: Visualizing the data ----------------------------------
# Plot some sample digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.show()

# -------------------------------- STEP 3: Splitting the data into training and testing sets --------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------- STEP 4: Initializing and training the SVM model ----------------------------------
# Initialize the SVM model
svm_model = SVC(kernel='linear', C=1)

# Train the SVM model on the training data
svm_model.fit(X_train, y_train)

# ---------------------------------- STEP 5: Testing the model ----------------------------------
# Predict the labels for the test set
y_pred = svm_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
