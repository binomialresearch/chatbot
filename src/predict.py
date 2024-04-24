# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_resources():
    model = load_model('../models/lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_question(model, tokenizer, question, max_length):
    seq = tokenizer.texts_to_sequences([question])
    print("Tokenized sequence:", seq)  # Debug tokenization
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    print("Padded sequence:", padded)  # Debug padding
    prediction = model.predict(padded)
    print("Model prediction (raw):", prediction)  # See raw softmax probabilities
    return np.argmax(prediction)

if __name__ == "__main__":
    model, tokenizer = load_resources()
    question = "Are you available Saturday?"
    predicted_label = predict_question(model, tokenizer, question, 8)  # Assuming max_length from training
    print("Predicted label:", predicted_label)
