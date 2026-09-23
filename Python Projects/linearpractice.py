### Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

### Import Data
df = pd.read_csv("linear_regression_dataset.csv", sep=";")

### Plot Data
plt.scatter(df.experience, df.salary)
plt.xlabel("Experience")
plt.ylabel("Salary")

### Import Linear Regression Function
from sklearn.linear_model import LinearRegression

### Linear Regression Model
linear_reg = LinearRegression()

### Reshape Arrays
x = df.experience.values.reshape(-1,1)
y = df.salary.values.reshape(-1,1)

### Linear Regression Fit
linear_reg.fit(x, y)

import numpy as np

### Intercept Point on the Y-Axis
b0 = linear_reg.predict([[0]])
### Intersection Point
b0_ = linear_reg.intercept_
### Slope
b1 = linear_reg.coef_

### salary = 1663 + 1138 * experience (y = b0 + b1*x)
trial = 1663.89519747 + 1138.34819698 * 10
print (trial)
### The predict command gives the corresponding answer for the desired x value using the equation above
print (linear_reg.predict([[11]]))

array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
### Create Scatter Plot
plt.scatter(x, y, color="green")

### Determines salary values corresponding to the desired experience list
y_head = linear_reg.predict(array.reshape(-1, 1))
### This creates a line according to our salary and experience matches
plt.plot(array, y_head, color="red")

plt.show()
