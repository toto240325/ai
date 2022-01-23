# TODO: Add import statements
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
df = bmi_life_data
# print(bmi_life_data.to_string()) 

# print(df.corr())
# # df.plot()

X = df["BMI"]
Y = df["Life expectancy"]

plt.scatter(X,Y)

#exit()
# bmi = bmi_life_data[:,1]
# print(bmi)

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[["BMI"]],bmi_life_data[["Life expectancy"]])


# laos_life_exp = bmi_life_model.predict( (21.07931).array.reshape(-1,1))
min = bmi_life_model.predict([[20]])
max = bmi_life_model.predict([[30]])

min = min[0][0]
max = max[0][0]

print(f'min : {min} - max : {max} ')
plt.plot([20,30],[min,max])




#print(f'laos_life_exp : {laos_life_exp}')
plt.show()
exit()

def myfit(x_values, y_values,x):
    model = LinearRegression()
    model.fit(x_values, y_values)
    return model.predict(x)



# plot the results

plt.figure()



X_min = X.min()
X_max = X.max()
counter = len(regression_coef)
for W, b in regression_coef:
    counter -= 1
    color = [1 - 0.92 ** counter for _ in range(3)]
    plt.plot([X_min, X_max],[X_min * W + b, X_max * W + b], color = color)
plt.scatter(X, y, zorder = 3)

values_list = [-3,-2.5,-2.2,-1.75,-1,-0.5,1, 1.5, 2, 3]
values = [ [x] for x in values_list ]
predictions = myfit(X,y, values)
#predict = predictions[0]
print(f'predict : {predictions}')
plt.scatter(values, predictions, zorder=5)
plt.show()



# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = None
