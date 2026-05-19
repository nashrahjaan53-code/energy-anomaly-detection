import numpy as np
X = np.array([1, 2, 3]).reshape(-1, 1)  
y = np.array([2, 4, 6])  
w = np.array([1.5]) 

def loss_function(y_pred, y_true):
    return (y_pred - y_true)**2

loss_1st_example = loss_function(X[0]*w, y[0])
print(f"Loss for 1st example: {loss_1st_example}")


def cost_function(X, y, w):
    total_loss = 0
    for x_i, y_i in zip(X, y):
        total_loss += loss_function(x_i*w, y_i)
    return total_loss / len(X)

cost = cost_function(X, y, w)
print(f"Cost (MSE): {cost}")

def objective_function(X, y, w, lambda_reg=0.1):
    mse = cost_function(X, y, w)
    l2_penalty = lambda_reg * np.sum(w**2)
    return mse + l2_penalty

objective = objective_function(X, y, w)
print(f"Objective (MSE + L2): {objective}")
print(f"  MSE part: {cost_function(X, y, w)}")
print(f"  L2 penalty part: {0.1 * np.sum(w**2)}")