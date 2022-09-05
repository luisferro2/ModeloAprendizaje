"""
This file contains the functions and procedure necessary to perform
the training and prediction of a linear regression, simple or multiple.

To run everything, please close the matplotlib plot windows as they
appear so the program can flow correctly and completely.

Author: Luis Ignacio Ferro Salinas A01378248
Last update: september 4th, 2022.

Techniques used:
- training and testing data split
- simple forward pass without activation
- Mini-batch stochastic gradient descent
- Mean squared error loss function

Current hyperparameters:
- Number of epochs
- learning rate
- relative batch size 

Future work:
- backpropagation for neural network
- k-fold cross validation
- activation functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_split_train_test(df, training_proportion):
    """ This function is to split dataframe into its training and testing components.
    
    It's assumed that this df has the target y variable as the last column
    
    Args: 
        df: DataFrame -> 
            It's the DataFrame that contains all of the data.
        training_proportion: float E [0, 1] -> 
            It's the percentage of training data as a float between 0 and 1
    """    
    n_rows = df.shape[0]
    
    n_training_rows = int(n_rows * training_proportion)
    
    y = list(df.columns)[-1]

    
    X_train = df.sample(n_training_rows).drop(y, axis=1)
    train_indices = X_train.index
    
    X_test = df.drop(train_indices, axis=0).drop(y, axis=1)
    
    y_train = df.iloc[train_indices][y]
    y_test = df.drop(train_indices, axis=0)[y]
    
    return X_train, X_test, y_train, y_test

def calc_MSE_loss_gradient(row, w, y):
    n_parameters = row.shape[0]
    gradient = []
    for i in range(n_parameters):
        row_remaining = np.concatenate([row[:i], row[i + 1:]])
        w_remaining = np.concatenate([w[:i], w[i + 1:]])
        gradient += [2 * row[i] * (row[i] * w[i] - y.loc[row.name] + \
            np.dot(row_remaining, w_remaining))]
    return np.array(gradient)

def perform_fwd_passes_and_calc_grads(X_train, mini_batch_proportion, 
        w, y_train):
    """ Simple method to perform many forward passes. 
    
    The model also calculates gradient for each of the passes.
    
    Args: 
        X_train: DataFrame -> Contains the information from which the
            mini-batch will be extracted.
        mini_batch_proportion: float E [0, 1]-> Percentage of the 
            training data to use before updating the weights
    """
    
    n_samples = int(mini_batch_proportion * X_train.shape[0])
    
    mini_batch = X_train.sample(n_samples)
    #mini_batch = X_train.iloc[:n_samples]
    indices = list(mini_batch.index)
    
    gradients = mini_batch.apply(calc_MSE_loss_gradient, raw=False,
        axis=1, args=(w, y_train.loc[indices]), result_type="expand")
    
    # I'm using the broadcasting trick here to replicate the weights
    # vector as many times as
    # there are samples in the mini-batch to perform those many
    # dot products.
    
    # Shape analysis
    # (samples, features) dot (features) = (samples)
    return np.dot(mini_batch, w), indices, gradients
    
def MSE(y, y_pred):
    return (y - y_pred) ** 2


# Testing my functions with simple linear regression 1 independent var.
score_df = pd.read_csv("score.csv")
score_df["x_0"] = 1
score_df = score_df[["x_0", "Hours", "Scores"]]
X_train, X_test, y_train, y_test = make_split_train_test(score_df, 0.8)
w = np.random.rand(score_df.shape[1] - 1)

# Graph of the initial random weights.
plt.plot(X_test, w[0] + w[1] * X_test)
plt.scatter(X_test["Hours"], y_test)
plt.title("Initial line with random weights.")
plt.xlabel("Hours studied")
plt.ylabel("Score obtained")
plt.show()

losses = []

for i in range(10):
    y_pred, indices, gradients = \
        perform_fwd_passes_and_calc_grads(X_train, 1, w, y_train)
    row = X_train.loc[0]
    errors = np.array(MSE(y_train.loc[indices], y_pred))
    loss = np.sum(errors) / errors.shape[0]
    losses += [loss]
    print(f"loss is {loss}")
    gradient = np.average(gradients, axis=0)
    w -= 0.01 * gradient

plt.plot(range(len(losses)), losses)
plt.title("Loss function value")
plt.xlabel("Epoch number")
plt.ylabel("MSE loss score")
plt.show()

# Plotting line with training data.
plt.plot(X_train, w[0] + w[1] * X_train)
plt.scatter(X_train["Hours"], y_train)
plt.title("Trained line on training data")
plt.xlabel("Hours studied")
plt.ylabel("Score obtained")
plt.show()

# Plotting line with testing data.
plt.plot(X_test, w[0] + w[1] * X_test)
plt.scatter(X_test["Hours"], y_test)
plt.title("Trained line on testing data")
plt.xlabel("Hours studied")
plt.ylabel("Score obtained")
plt.show()


# Multiple variable example, health insurance charges prediction
# based on the age, bmi and number of children.
insurance_df = pd.read_csv("insurance.csv")
insurance_df.drop(["sex", "smoker", "region"], axis=1, inplace=True)
insurance_df["x_0"] = 1
insurance_df = insurance_df[["x_0", "age", "bmi", 
                             "children", "charges"]]
X_train, X_test, y_train, y_test = \
    make_split_train_test(insurance_df, 0.8)

print("Training 3 variable model on dataset with 1300 instances for \
    300 epochs...")

w = np.random.rand(insurance_df.shape[1] - 1)

losses = []

for i in range(300):
    print(F"EPOCH {i}")
    y_pred, indices, gradients = \
        perform_fwd_passes_and_calc_grads(X_train, 0.8, w, y_train)
    row = X_train.loc[0]
    errors = np.array(MSE(y_train.loc[indices], y_pred))
    loss = np.sum(errors) / errors.shape[0]
    losses += [loss]
    #print(f"loss is {loss}")
    gradient = np.average(gradients, axis=0)
    #gradient
    #print(f"gradient is {gradient}")
    #print(f"w before {w}")
    gradient_norm = np.linalg.norm(gradient)
    #print(f"gradient norm is {np.linalg.norm(gradient)}")
    if loss > 1000 and gradient_norm < 400: 
        learning_rate = 0.05
    else:
        learning_rate = 0.0003
    w -= learning_rate * gradient
    #print(f"w after {w}")

print("DONE!")
# Plotting loss for multivariate example.
plt.plot(range(len(losses)), losses)
plt.title("Loss function")
plt.xlabel("Epoch number")
plt.ylabel("MSE loss score")
plt.show()

age = int(input("Please enter your age: "))
bmi = float(input("Please enter your bmi: "))
children = int(input("Please enter your numnber of children: "))

print(f"Your health insurance premium charges will be  \
    {np.dot(w, np.array([1, age, bmi, children]))}")






