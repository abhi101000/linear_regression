# Task 3: Linear Regression - Elevate Labs AI & ML Internship
# Author: Abhishek Pandey
# Objective: Implement simple & multiple linear regression on Housing Price dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("Housing.csv")  

print("Dataset Shape:", df.shape)
print(df.head())

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

feature_name = X.columns[0]
plt.scatter(X_test[feature_name], y_test, color="blue", label="Actual")
plt.scatter(X_test[feature_name], y_pred, color="red", label="Predicted")
plt.xlabel(feature_name)
plt.ylabel("Price")
plt.title("Simple Linear Regression Example")
plt.legend()
plt.show()

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients)
print("Intercept:", model.intercept_)


plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
