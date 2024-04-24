# preprocess.py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
    questions, labels = zip(*(line.split(', ') for line in lines))
    return list(questions), list(map(int, labels))

def preprocess_data(questions):
    tokenizer = Tokenizer(num_words=80, oov_token="<OOV>")
    tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    max_length = max(len(x) for x in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    with open('tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return padded_sequences, max_length

if __name__ == "__main__":
    questions, labels = load_data('../data/questions_and_labels.csv')
    padded_sequences, max_length = preprocess_data(questions)
    np.save('padded_sequences.npy', padded_sequences)
    np.save('labels.npy', np.array(labels))
    # Printing the variables
    print("Questions:", questions)
    print("Labels:", labels)
    print("Padded Sequences:\n", padded_sequences)
    print("Max Length of Sequences:", max_length)
