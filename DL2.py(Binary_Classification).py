import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D

# Load IMDB dataset 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) 
X_train = pad_sequences(X_train, maxlen=200) 
X_test = pad_sequences(X_test, maxlen=200) 

# Define model 
model = Sequential([ 
Embedding(10000, 32), 
GlobalAveragePooling1D(), 
Dense(1, activation='sigmoid') 
]) 

# Compile and train model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test)) 

# Evaluate model 
test_loss, test_acc = model.evaluate(X_test, y_test) 
print(f"Test Accuracy: {test_acc}")