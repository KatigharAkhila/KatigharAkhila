import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.initializers import Orthogonal

def load_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    # Add other layers to your model as needed
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
