"""
train_model.py
----------------
Trains an LSTM next-word-prediction model on a text corpus and saves:
  - next_word_model.h5   (the trained Keras model)
  - tokenizer.pickle      (the fitted tokenizer, needed at inference time)
  - model_config.pickle   (max_sequence_len, vocab_size — needed by the app)

Usage:
    python train_model.py                 # trains on corpus.txt
    python train_model.py my_text.txt     # trains on your own text file
"""

import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

CORPUS_PATH = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
EPOCHS = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 150

def main():
    # 1. Load text
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    corpus = [line.strip() for line in text.split("\n") if line.strip()]
    print(f"Loaded {len(corpus)} lines from {CORPUS_PATH}")

    # 2. Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {total_words}")

    # 3. Build n-gram training sequences from every line
    #    e.g. "the sun was setting" -> [the,sun], [the,sun,was], [the,sun,was,setting]
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[: i + 1])

    max_sequence_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding="pre"
    )
    print(f"Total training sequences: {len(input_sequences)}")
    print(f"Max sequence length: {max_sequence_len}")

    # 4. Split into X (context) / y (next word)
    X = input_sequences[:, :-1]
    labels = input_sequences[:, -1]
    y = to_categorical(labels, num_classes=total_words)

    # 5. Build the LSTM model
    model = Sequential([
        Embedding(total_words, EMBEDDING_DIM, input_length=max_sequence_len - 1),
        LSTM(LSTM_UNITS),
        Dense(total_words, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # 6. Train
    model.fit(X, y, epochs=EPOCHS, verbose=1)

    # 7. Save everything the Streamlit app will need
    # .keras is the modern native format (replaces the legacy .h5 format)
    model.save("next_word_model.keras")
    with open("tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("model_config.pickle", "wb") as f:
        pickle.dump({"max_sequence_len": max_sequence_len, "total_words": total_words}, f)

    print("\nSaved: next_word_model.keras, tokenizer.pickle, model_config.pickle")


if __name__ == "__main__":
    main()