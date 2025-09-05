# Week 1 - Sustainable Agriculture Project
# Theme: Applying AI to improve Green Skills

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Step 1: Load the dataset
print("Loading the dataset...")
data = pd.read_csv("sustainable_agriculture.csv")
print("\nHere are the first 5 rows of our data:")
print(data.head())
# Step 2: Check for missing values
print("\nChecking if there are any missing values in the dataset...")
print(data.isnull().sum())
# Step 3: Get basic statistics
print("\nSome basic statistics about the dataset:")
print(data.describe())
# Step 4: Visualize the data to understand trends
# Plotting Rainfall vs Crop Yield
plt.scatter(data["Rainfall_mm"], data["Crop_Yield_tons_per_hectare"])
plt.xlabel("Rainfall (mm)")
plt.ylabel("Crop Yield (tons/hectare)")
plt.title("How Rainfall affects Crop Yield")
plt.show()
# Plotting Fertilizer vs Crop Yield
plt.scatter(data["Fertilizer_kg_per_hectare"], data["Crop_Yield_tons_per_hectare"])
plt.xlabel("Fertilizer (kg/hectare)")
plt.ylabel("Crop Yield (tons/hectare)")
plt.title("How Fertilizer affects Crop Yield")
plt.show()
# Step 5: Build a simple linear regression model to see the effect of fertilizer on yield
X = data[["Fertilizer_kg_per_hectare"]]
y = data["Crop_Yield_tons_per_hectare"]
model = LinearRegression()
model.fit(X, y)
print("\nLinear Regression Results:")
print("Coefficient (how much yield changes per kg of fertilizer):", model.coef_)
print("Intercept (expected yield with zero fertilizer):", model.intercept_)
