import pandas as pd

df = pd.read_csv(r"D:\VS CODE\python\ROP_data .csv")   # updated path

print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nFirst 5 Rows:\n", df.head())

from sklearn.model_selection import train_test_split

# Select input features (X) and target (y)
X = df[["Depth", "WOB", "SURF_RPM", "PHIF", "VSH", "SW", "KLOGH"]]
y = df["ROP_AVG"]

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

print("Linear Regression Model Trained Successfully!")
print("First 5 Predictions:", y_pred[:5])

import matplotlib.pyplot as plt

# Sort test indices so graph looks clean (like your example)
X_test_sorted = X_test.copy()
X_test_sorted["y_test"] = y_test
X_test_sorted["y_pred"] = y_pred
X_test_sorted = X_test_sorted.sort_values(by="Depth")

# Extract sorted actual & predicted values
actual = X_test_sorted["y_test"].values
pred = X_test_sorted["y_pred"].values

# Plot
plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual ROP", color="red")
plt.plot(pred, label="Linear Regression", linestyle="--", marker="o")
plt.title("Actual vs Linear Regression (ROP Prediction)")
plt.xlabel("Samples (Sorted by Depth)")
plt.ylabel("ROP (m/h)")
plt.legend()
plt.grid(True)
plt.show()


