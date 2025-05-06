import numpy as np

def svm_train(X, y, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
    """
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values (should be -1 or 1)
    learning_rate : float, default=0.01
        Step size for gradient descent
    lambda_param : float, default=0.01
        Regularization parameter
    num_iterations : int, default=1000
        Number of iterations for gradient descent
    
    Returns:
    --------
    weights : array-like, shape (n_features,)
        Learned weights of the model
    bias : float
        Learned bias term of the model
    costs : list
        Cost history during training
    """
    # Make sure y contains -1 and 1
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y should contain only -1 and 1 values")
    
    # Get dimensions of X
    m, n = X.shape  # m = number of examples, n = number of features
    
    # Initialize parameters
    weights = np.zeros(n)
    bias = 0
    costs = []
    
    # Gradient descent
    for i in range(num_iterations):
        # Calculate margin values: yi(w·xi + b)
        margins = y * (np.dot(X, weights) + bias)#y is vector of output column
        # If margin ≥ 1 the point is correct & outside the margin.
        # Calculate cost (hinge loss + regularization)
        # max(0, 1 - margin) for each example
        hinge_loss = np.maximum(0, 1 - margins)
        cost = lambda_param/2 * np.dot(weights, weights) + np.sum(hinge_loss) / m
        
        # Initialize gradients
        dw = lambda_param * weights
        db = 0
        
        # Update gradients for misclassified points (where margin < 1)
        misclassified = margins < 1#This will be a matrix
        dw -= np.dot(X[misclassified].T, y[misclassified]) / m
        db -= np.sum(y[misclassified]) / m
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Save cost
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    
    return weights, bias, costs

def svm_predict(X, weights, bias):
    
    # Calculate the decision function
    decision = np.dot(X, weights) + bias
    
    # Return the sign of the decision function
    return np.sign(decision)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
