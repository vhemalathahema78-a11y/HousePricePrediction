import pandas as pd

# Load the house dataset
data = pd.read_csv("house_data.csv")

# Display the first five rows
print("House Price Prediction Project")
print(data.head())

# Display dataset information
print("\nTotal houses in dataset:", len(data))
print("\nColumn names:")
print(data.columns.tolist())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("house_data.csv")

# Select input features and target price
X = data[["area_sqft", "bedrooms", "bathrooms"]]
y = data["price_lakhs"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices for test data
y_pred = model.predict(X_test)

# Evaluate the model
print("House Price Prediction Model")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Predict the price of a new house
new_house = pd.DataFrame({
    "area_sqft": [1500],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(new_house)

print("\nNew House Details:")
print("Area: 1500 sqft")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Predicted Price:", round(predicted_price[0], 2), "lakhs")