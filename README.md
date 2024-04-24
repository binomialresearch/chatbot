# LSTM Chatbot

This project contains an LSTM-based chatbot that classifies input questions into categories and provides predefined responses. The chatbot is trained on a small dataset and can handle slight variations in question phrasing.

## Project Structure

This repository is organized as follows:

- `data/`: Contains the dataset files.
- `models/`: Where trained models and tokenizer data are saved.
- `src/`: Source code for the chatbot including preprocessing, training, and prediction scripts.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.x installed on your system. You also need the following Python packages:

- TensorFlow
- NumPy

You can install them using pip:

```bash
pip install tensorflow numpy

cd src
python preprocess.py
python train.py
python predict.py
