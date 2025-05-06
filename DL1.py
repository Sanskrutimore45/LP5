import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset from CSV
df = pd.read_csv('BostonHousing.csv')  # Ensure this file is in the same directory

# Step 2: Separate independent (X) and dependent (y) variables
X = df.drop('medv', axis = 1)  # MEDV is the target: Median value of owner-occupied homes
y = df ['medv']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 5: Define the Deep Neural Network (DNN) model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Step 6: Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss = 'mse',
              metrics=['mae'])
# Mean Absolute Error


#Step 7: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Step 8: Evaluate the model
test_mae = model.evaluate(X_test, y_test)[1]  # Index 1 = 'mae'


# Step 9: Predict on the first 5 test examples
y_pred = model.predict(X_test[:5]).flatten()


# Step 10: Print the results
print(f"Test MAE: {test_mae}")
print("Actual Prices:", y_test[:5].values)
print("Predicted Prices:", y_pred)










#used when asked for accuracy and all...


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Predict on test data
y_pred_full = model.predict(X_test).flatten()

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred_full))

# R^2 Score (like "accuracy" for regression)
r2 = r2_score(y_test, y_pred_full)

# Error rate can be considered as relative error
relative_error = np.mean(np.abs((y_test - y_pred_full) / y_test)) * 100

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")
print(f"Relative Error Rate (%): {relative_error:.2f}%")

