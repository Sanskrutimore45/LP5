import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Step 1: Load and split the dataset (Independent variables X, Dependent variable y)
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Step 2: Standardize the data to improve model performance
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Step 3: Define the Deep Neural Network (DNN) model with enhancements
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Increased neurons
    Dropout(0.2),  # Dropout to prevent overfitting
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Linear output layer for predicting continuous values
])

# Step 4: Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Step 5: Train the model using Backpropagation with increased epochs and different batch size
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Step 6: Evaluate model performance on test data
test_mae = model.evaluate(X_test, y_test)[1]  # MAE is the second value returned

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test[:5]).flatten()  # Predict the first 5 examples from the test set

# Step 8: Display results
print(f"Test MAE: {test_mae}")
print("Actual Prices:", y_test[:5])
print("Predicted Prices:", y_pred)
