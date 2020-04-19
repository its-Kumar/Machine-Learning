# Polynomial Regression model

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# fitting linear reg model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) 

# fitting pplynomial reg. model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 =LinearRegression()
lin_reg2.fit(X_poly,Y)

# Visualising the linear reg result
plt.scatter(X, Y, color="red")
plt.plot(X,lin_reg.predict(X), color="blue")
plt.title("Truth or Bluff(Linear Reg.)")
plt.xlabel("Position Lavel")
plt.ylabel('Salary')
plt.show()


# Visiuling polynomial reg result
X_grid =np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color="red")
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color="blue")
plt.title("Truth or Bluff(Polynomial Reg.)")
plt.xlabel("Position Lavel")
plt.ylabel('Salary')
plt.show()

# predicting new result with linear reg
lin_reg.predict([[6.5]])

# predicting new result with Polynomial reg
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
