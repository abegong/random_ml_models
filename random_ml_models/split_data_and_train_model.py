import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

def split_data_and_train_model(
    data,
    model=LinearRegression(),
    train_proportion=.8,
):
    # Split into X and y, using the last variable as the dependent variable
    X = data[:,:-1]
    y = data[:,-1]

    # Create a random variable for an 80/20 train/test split.
    z = np.random.uniform(size=y.shape[0]) < .8

    X_train = X[z,:]
    y_train = y[z]
    X_test = X[z==False,:]
    y_test = y[z==False]

    # Train model
    model.fit(X_train, y_train)

    return {
        "X_train": X_train,
        "Y_train": y_train,
        "X_test": X_test,
        "Y_test": y_test,
        "model": model,
    }