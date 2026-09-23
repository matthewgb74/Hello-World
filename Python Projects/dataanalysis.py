import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# the following is a sample of methods for analyzing date in python
#
# ==========================================================
# input output sample data
# ==========================================================
x = np.arange(0,20,2)
y = np.array([2, 4, 5, 13, 17, 15, 16, 20, 19, 25])

# ==========================================================
# polynomial interpolation
# ==========================================================
degree = 5

# find the coefficients of the polynomial
coeffs = np.polyfit(x, y, degree)

# create the polynomial with the coefficients you found
p = np.poly1d(coeffs)

# linearly space the x values
x_new = np.linspace(x[0], x[-1], 200)

# evaluate the polynomial at each of the new x vals
y_new = p(x_new)

# new line and title for the output
print("\n=== Polynomial Interpolation ===")
#
print("Polynomial coefficients: ", np.round(coeffs, 3))


# ==========================================================
# exponential regression w/ curve_fit()
# ==========================================================
# define a function that takes P,r,t as input and outputs the
#   exponential function E = P*e^(rt)
def exponential_model(t, P, r):
    return P * np.exp(r * t)

# Use curve_fit() to find the P,r and error
params, cov_matrix = curve_fit(exponential_model, x, y, p0=[1, 0.1])

# extract a,b from params
P, r = params

# create outputs of the exponential model
y_exp = exponential_model(x_new, P, r)

print("\n=== Nonlinear Regression (Exponential) ===")
print(f"Model: y = {P:.3f} * e^({r:.3f}t)")

# the diagonals of the covarience matrix are the standard deviation
std_P, std_r = np.sqrt(np.diag(cov_matrix))
print(f"P = {params[0]:.3f} ± {std_P:.3f}")
print(f"r = {params[1]:.3f} ± {std_r:.3f}")


# ==========================================================
# calculating r^2 (polynomial)
# ==========================================================
# indeces for the 10 data points we need
idx_r2 = np.linspace(0, len(x_new)-1, len(x)).astype(int)

# y values for the polynomial function at the needed x inputs
y_comp_p = y_new[idx_r2]

corr_matrix_p = np.corrcoef(y,y_comp_p)
corr_p = corr_matrix_p[0,1]
r_sq_p = corr_p**2
print("\n=== r^2 (polynomial) ===")
print(f"r^2 = {r_sq_p:.3}")

# ==========================================================
# calculating r^2 (exponential)
# ==========================================================
# y values for the exponential function at the needed x inputs
y_comp_e = y_exp[idx_r2]

corr_matrix_e = np.corrcoef(y,y_comp_e)
corr_e = corr_matrix_e[0,1]
r_sq_e = corr_e**2
print("\n=== r^2 (exponential) ===")
print(f"r^2 = {r_sq_e:.3}")


# ==========================================================
# polynomial differentiation
# ==========================================================
#name a function for finding the derivatives of the polynomial
dp = p.deriv()

# find the derivative at every value stored in x_new
dydx = dp(x_new)

# choose a specific value to find the derivative
x_valP = 2.5

print("\n=== Polynomial Derivative ===")
print(f"The derivative of p at x = {x_valP} is p'({x_valP}) = {dp(x_valP):.3}")

# ==========================================================
# numerical differentiation (on the exponential)
# ==========================================================
# computes m = (y2-y1)/(x2-x1) for consecutive elements
dy_dx = np.diff(y_exp) / np.diff(x_new)

# takes the avergae of the consecutive values to match dy_dx
x_mid = (x_new[:-1] + x_new[1:]) / 2

# choose a specific value of to find the derivative
x_valE = 2.5

# finds the index for the cloesst number to x_valE
idx = np.argmin(np.abs(x_mid - x_valE))

print("\n===Numerical Derivative (exponential)===")
print(f"The derivative of E at x = {x_valE} is E'({x_valE}) = {dy_dx[idx]:.3}")

# ==========================================================
# polynomial integration
# ==========================================================
# create a function for finding the integral
P = p.integ()

# bounds
a = 0
b = 10

# FTOC pt 2
area = P(b) - P(a)

print("\n=== Integration ===")
print(f"Integral of polynomial from x = {a} to x = {b}: {area}")


# ==========================================================
# trapazoid integration
# ==========================================================
# create a function for trapezioidal rule
area = np.trapezoid(y,x)

print(f"The area approx using trapezoids is {area}.")

# ==========================================================
# statistical functions on y
# ==========================================================
print("\n=== Statistics of y ===")
print("Mean:", np.mean(y))
print("Median:", np.median(y))
print("Standard deviation:", np.std(y))
# ==========================================================
# plotting
# ==========================================================
# choose display size of the graph
plt.figure(figsize=(10,6))

# OG data
plt.scatter(x, y, color="black", label="Original Data Points")

# polynomial interpolation
plt.plot(x_new, y_new, label=f"Degree {degree} Polynomial Interpolation", linewidth=2)

# exponential curve fitting
plt.plot(x_new, y_exp, ":", label="Exponential Regression")

# polynomial derivative
plt.plot(x_new, dydx, label="Polynomial Derivative")

# numerical derivative
plt.plot(x_mid, dy_dx, ":", label="Numerical Derivative")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolation and Regression")
plt.legend()
plt.grid(True)
plt.show()