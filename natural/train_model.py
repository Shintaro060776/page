import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

df = pd.read_csv('normalized_jokes.csv')
jokes = df['Normalized Joke'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(jokes)
sequences = tokenizer.texts_to_sequences(jokes)
max_sequence_len = max([len(x) for x in sequences])
sequences = np.array(pad_sequences(
    sequences, maxlen=max_sequence_len, padding='pre'))

X = sequences[:, :-1]
labels = sequences[:, -1]
vocabulary_size = len(tokenizer.word_index) + 1
Y = tf.keras.utils.to_categorical(labels, num_classes=vocabulary_size)

model = Sequential()
model.add(Embedding(vocabulary_size, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(vocabulary_size, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=64)

model.save('joke_generator_model.h5')
