import numpy as np

def linear_regression(X, y, learning_rate=0.01, num_iterations=1000):
    """
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
        # Forward pass: calculate predictions (y_hat = wx + b)
        predictions = np.dot(X, weights) + bias
        
        # Calculate error
        error = predictions - y
        
        # Calculate cost (mean squared error)
        cost = (1/(2*m)) * np.sum(error**2)
        
        # Calculate gradients
        dw = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Save cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    
    return weights, bias, costs

def predict(X, weights, bias):
    """
    Make predictions using the trained linear regression model
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    weights : array-like, shape (n_features,)
        Model weights
    bias : float
        Model bias
    
    Returns:
    --------
    array-like, shape (n_samples,)
        Predicted values
    """
    return np.dot(X, weights) + bias

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 examples, 1 feature
    y = 2 * X.reshape(-1) + 1 + np.random.randn(100) * 1  
    
    # Train the model
    weights, bias, costs = linear_regression(X, y, learning_rate=0.01, num_iterations=1000)
    
    # Print results
    print(f"Weights: {weights}, Bias: {bias}")
    print(f"Equation: y = {weights[0]:.4f}x + {bias:.4f}")
    
    # Make predictions
    y_pred = predict(X, weights, bias)
    
    # Calculate R-squared (coefficient of determination)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    print(f"R-squared: {r_squared:.4f}")
    