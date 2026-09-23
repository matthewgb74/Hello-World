### Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

### Import Data
df = pd.read_csv("medianhomeprice.csv", sep=",")

### Convert observation_date to datetime, then extract year as numeric
df["observation_date"] = pd.to_datetime(df["observation_date"])
df["year"] = df["observation_date"].dt.year

### Group by year and take the mean price
df_yearly = df.groupby("year", as_index=False)["price"].mean()

### Filter data to only include years >= 1984
df_yearly = df_yearly[df_yearly["year"] >= 1984]

### Print Columns to Make Sure Formatting is Correct
print(df.columns)

### Plot Data
plt.scatter(df_yearly["year"], df_yearly["price"], color="black", s=10)
plt.xlabel("Year")
plt.ylabel("Median Home Price")
plt.title ("US Median Home Price")

### Import Linear Regression Function
from sklearn.linear_model import LinearRegression

### Linear Regression Model
linear_reg = LinearRegression()

### Reshape Arrays
x = df_yearly["year"].values.reshape(-1,1)
y = df_yearly["price"].values.reshape(-1,1)

### Linear Regression Fit
linear_reg.fit(x, y)

intercept = linear_reg.intercept_[0]  # scalar value
slope = linear_reg.coef_[0][0]        # scalar value
trial = intercept + slope * 2030      # y = b0 + b1*x
print(f"Predicted Home Price in 2030: ${trial:.2f}")

x_line = np.arange(df_yearly["year"].min(), df_yearly["year"].max() + 1).reshape(-1, 1)
y_line = linear_reg.predict(x_line)
plt.plot(x_line, y_line, color="red")

### Polynomial Interpolation
degree = 5

### Create variables for 1D Array or you can use flatten function
x_poly = df_yearly["year"].values
y_poly = df_yearly["price"].values

### Coefficients
coeffs = np.polyfit(x_poly, y_poly, degree)
print(f"Polynomial coefficients: {coeffs}")

### Create Polynomial Function
polynomial = np.poly1d(coeffs)

### Linearly Space Points
x_new = np.linspace(min(x), max(x), 200) # Create a smooth range of x values
y_fitted = polynomial(x_new)

### Predicted values using the polynomial
y_pred = polynomial(x)  
predicted_2030 = polynomial(2030)

### R-Squared Function
r2linear = linear_reg.score(x, y)
r2poly = r2_score(y, y_pred) 

### This creates a line according to our Year and Income matches
plt.plot(x_line, y_line, color="red", label=f"Linear Regression, Line R^2 = {r2linear:.2f}")
plt.plot(x_new, y_fitted, color="blue", label=f"Fitted Polynomial (Degree {degree}), R^2 = {r2poly:.2f}")
plt.legend()
plt.show()

### Basic Statistics
print("\n=== Basic Statistics ===")
print("Mean:", np.mean(y))
print("Median:", np.median(y))
print("Standard deviation:", np.std(y))
print(f"Linear Regression R-Squared: {r2linear:.4f}")
print(f"Polynomial R-Squared: {r2poly:.4f}")

### Extrapolations
print("\n=== Predictions ===")
print(f"Linear Predicted Home Price in 2030 is ${trial:.2f}")
print(f"Polynomial Predicted Home Price in 2030 is ${predicted_2030:.2f}")


