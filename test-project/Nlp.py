import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sample text data
sentences = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The bird flew to the tree"
]
# Tokenizing and padding text
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')
# Neural network model
model = Sequential()
model.add(Embedding(input_dim=100, output_dim=64))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
# Compile and summarize the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()