# train.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import pickle
from tensorflow.keras.optimizers import Adam

def load_data():
    padded_sequences = np.load('padded_sequences.npy')
    labels = np.load('labels.npy')
    print("Label distribution:", {label: np.sum(labels == label) for label in set(labels)})
    return padded_sequences, labels

def build_model(input_length, num_classes):
    model = Sequential([
        Embedding(input_dim=200, output_dim=8, input_length=input_length),
        LSTM(8),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.1)  # Increase learning rate here
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

if __name__ == "__main__":
    padded_sequences, labels = load_data()
    input_length = padded_sequences.shape[1]
    num_classes = len(set(labels))
    print('input length', input_length, 'num classes', num_classes)
    model = build_model(input_length, num_classes)
    model.fit(padded_sequences, labels, epochs=5, validation_split=0.2)
    model.summary()
    model.save('../models/lstm_model.h5')
