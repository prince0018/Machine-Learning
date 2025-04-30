# The likelihood function is non-linear due to the logistic function, and there is no analytical solution for the maximum likelihood estimates.
#cross entropy comes from bernoulli distribution, which is the likelihood function for binary classification problems.
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Logistic Regression implementation using gradient descent from scratch
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data
    y : array-like, shape (n_samples,)
        Target values (0 or 1)
    learning_rate : float, default=0.01
        Step size for gradient descent
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
    # Get dimensions of X
    m, n = X.shape  # m = number of examples, n = number of features
    
    # Initialize parameters
    weights = np.zeros(n)
    bias = 0
    costs = []
    
    # Gradient descent
    for i in range(num_iterations):
        # Forward pass: calculate predictions
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        # Calculate cost (log loss)
        cost = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Calculate gradients
        dw = 1/m * np.dot(X.T, (predictions - y))
        db = 1/m * np.sum(predictions - y)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Save cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    
    return weights, bias, costs

def predict(X, weights, bias, threshold=0.5):
    """
    Predict class labels for input samples
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    weights : array-like, shape (n_features,)
        Model weights
    bias : float
        Model bias
    threshold : float, default=0.5
        Decision threshold
    
    Returns:
    --------
    array-like, shape (n_samples,)
        Predicted class labels (0 or 1)
    """
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return (probs >= threshold).astype(int)

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 examples, 2 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear boundary
    
    # Train the model
    weights, bias, costs = logistic_regression(X, y, learning_rate=0.1, num_iterations=1000)
    
    # Make predictions
    y_pred = predict(X, weights, bias)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.4f}")