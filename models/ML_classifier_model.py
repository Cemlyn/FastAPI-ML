# Code source: Jaques Grobler
# License: BSD 3 clause
import pickle
import sys
sys.path.append('..')

import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



class CustomMLmodel:
    def __init__(self,model):
        self._model = model

    def fit(self,X,y):
        self._model.fit(X,y)

    def predict(self,X):
        return self._model.predict(X)

if __name__ == "__main__":
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    model1 = CustomMLmodel(model=linear_model.LinearRegression())

    # Train the model using the training sets
    model1.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = model1.predict(diabetes_X_test)

    # The coefficients
    print("Coefficients: \n", model1._model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

    with open('./assets/model1.pickle','wb') as file:
        pickle.dump(model1,file)

    # Create linear regression object
    model2 = CustomMLmodel(model=linear_model.HuberRegressor())

    # Train the model using the training sets
    print("Diabetes shape:",diabetes_X_train.shape)
    model2.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = model2.predict(diabetes_X_test)

    # The coefficients
    print("Coefficients: \n", model2._model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

    with open('./assets/model2.pickle','wb') as file:
        pickle.dump(model2,file)
