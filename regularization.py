# TODO: Add import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data_regul.csv', header = None)
X = train_data.iloc[:,0]
y = train_data.iloc[:,-1]

plt.scatter(X,y)

linear_model = LinearRegression()
linear_model.fit(train_data[[0]],train_data[[-1]])

mypredict = linear_model.predict([[0]])
print(mypredict)


plt.show()

exit()




# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
