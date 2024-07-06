# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Problem Statement.....
# You have given a dataset that describes the house in California. Now, based on the given features, you have to predict the house price.

# Creating a Dataframe
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)

# EDA - Exploratory Data Analysis(grabs all info about data)

# Adding the target column
df["Price"] = california.target

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.nunique())
print(df.isnull().sum())
print(df.describe())

plt.figure(figsize=(20,20))
sns.heatmap(data=df.corr(), annot=True, cmap="Greens")
plt.show()

sns.pairplot(df, height=5)
plt.show()

df.boxplot(figsize=(10,10))
plt.show()

# Export the dataset
df.to_csv('California_dataset.csv')

print(np.min(df.Price))
print(np.max(df.Price))
print(np.std(df.Price))

x1 = np.array(df.drop("Price", axis=1))
y1 = np.array(df.Price)
x = california.data
y = california.target
print(x)
print(y)

# Splitting the data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

# Choosing the model
model = LinearRegression()

# Train/Model the model
model.fit(x_train, y_train)

# Intercept value
print(model.intercept_)
# Coefficient value
print(model.coef_)

# Prediction/Testing the model
print(y_test)
y_predict = model.predict(x_test)
print(y_predict)

# Testing the model performance
print(model.score(x_test, y_test))

# R squared
print(r2_score(y_test, y_predict))

# MSE
print(mean_squared_error(y_test, y_predict))

# MAE
print(mean_absolute_error(y_test, y_predict))

plt.scatter(y_test, y_predict)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price v/s Predicted price")
plt.grid()
plt.plot([min(y_test), max(y_test)],[min(y_predict), max(y_predict)], color = "red")
plt.show()
