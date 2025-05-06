import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import SimpleRNN, Dense 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
import kagglehub 

# Download dataset using kagglehub 
path = kagglehub.dataset_download("akram24/google-stock-price-test") 
dataset_path = f"{path}/Google_Stock_Price_Test.csv" 

# Load dataset 
data = pd.read_csv(dataset_path, usecols=[1]).dropna().values.astype(float) 

# Normalize data 
scaler = MinMaxScaler() 
data = scaler.fit_transform(data) 

# Prepare time series data 
def create_dataset(dataset, time_step=10): 
    X, y = [], [] 
    for i in range(len(dataset) - time_step): 
        X.append(dataset[i:i + time_step, 0]) 
        y.append(dataset[i + time_step, 0]) 
    return np.array(X), np.array(y) 

# Set time step
time_step = min(10, len(data) - 1)

# Check if enough data is available
if len(data) > time_step:
    X, y = create_dataset(data, time_step) 
    X = X.reshape(X.shape[0], X.shape[1], 1) 

    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # Define RNN model 
    model = Sequential([ 
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(time_step, 1)), 
        SimpleRNN(50, activation='relu'), 
        Dense(1) 
    ]) 

    # Compile and train model 
    model.compile(optimizer='adam', loss='mse') 
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)) 

    # Evaluate model 
    test_loss = model.evaluate(X_test, y_test) 
    print(f"Test Loss: {test_loss}")
else: 
    print("Error: Not enough data points to create sequences. Consider using a smaller time step.")
