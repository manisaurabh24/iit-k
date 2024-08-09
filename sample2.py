import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers

# Load the dataset
data = pd.read_csv('pulsar_star_dataset.csv')

# Split the data into features and target variable
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values  # The last column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)

def svm_train(X, y, C):
    m, n = X.shape
    y = y.astype(float)
    y = y * 2 - 1  # Transform labels from {0, 1} to {-1, 1}
   
    # Compute the kernel matrix
    K = linear_kernel(X, X)
   
    # Set up the parameters for the quadratic programming problem
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(m))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = matrix(y, (1, m), 'd')
    b = matrix(0.0)
   
    # Solve the quadratic programming problem
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
   
    # Extract the Lagrange multipliers
    alphas = np.array(solution['x']).flatten()
   
    # Support vectors have non zero lagrange multipliers
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    alphas = alphas[sv]
    sv_X = X[sv]
    sv_y = y[sv]
   
    # Calculate the intercept
    b = np.mean(sv_y - np.sum(alphas * sv_y[:, None] * K[ind][:, sv], axis=1))
   
    # Weight vector for the linear kernel
    w = np.sum(alphas[:, None] * sv_y[:, None] * sv_X, axis=0)
   
    return w, b

def svm_predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

def evaluate_svm(C_values):
    results = {}
    for C in C_values:
        w, b = svm_train(X_train, y_train, C)
        predictions = svm_predict(X_test, w, b)
        accuracy = np.mean(predictions == (y_test * 2 - 1))
        results[C] = accuracy
        print(f"Accuracy for C={C}: {accuracy}")
    return results

# Hyperparameter values for C
C_values = [0.1, 1, 10, 100, 1000]

# Evaluate the SVM for each value of C
results = evaluate_svm(C_values)
