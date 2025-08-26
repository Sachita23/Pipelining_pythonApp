import os
# Disable GPU usage to avoid CUDA errors on environments without GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def train_model():
    # Load dataset
    df = pd.read_csv('data.csv')
    X = df[['Feature1', 'Feature2']].values
    y = df[['Output']].values

    # Define model
    model = Sequential()
    model.add(Input(shape=(2,)))           # Explicit Input layer
    model.add(Dense(32, activation='linear'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))                    # Output layer

    # Compile and train
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=1000, verbose=0)

    return model
