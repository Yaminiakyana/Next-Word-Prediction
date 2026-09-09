"""
app.py
-------
Streamlit app: type some text, and an LSTM model predicts the next word
(or generates several words ahead).

Run locally with:
    streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="LSTM Next Word Predictor", page_icon="🌀", layout="centered")


@st.cache_resource
def load_artifacts():
    model = load_model("next_word_model.keras")
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    with open("model_config.pickle", "rb") as f:
        config = pickle.load(f)
    return model, tokenizer, config


def predict_next_words(seed_text, model, tokenizer, max_sequence_len, num_words=1, top_k=1):
    """
    Predicts the next `num_words` words after `seed_text`.
    Returns the generated text and, for the very first prediction,
    the top_k candidate words with their probabilities.
    """
    text = seed_text
    top_candidates = []

    for step in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)[0]

        if step == 0:
            top_indices = predicted_probs.argsort()[-top_k:][::-1]
            index_to_word = {v: k for k, v in tokenizer.word_index.items()}
            top_candidates = [
                (index_to_word.get(i, "?"), float(predicted_probs[i])) for i in top_indices
            ]

        predicted_index = int(np.argmax(predicted_probs))
        index_to_word = {v: k for k, v in tokenizer.word_index.items()}
        next_word = index_to_word.get(predicted_index, "")
        if not next_word:
            break
        text += " " + next_word

    return text, top_candidates


# ---------------- UI ----------------

st.title("LSTM Next Word Predictor")
st.write(
    "Type a phrase and the LSTM model will predict what word (or words) "
    "come next, based on the text it was trained on."
)

try:
    model, tokenizer, config = load_artifacts()
    max_sequence_len = config["max_sequence_len"]
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_model.py` first to generate "
        "`next_word_model.keras`, `tokenizer.pickle`, and `model_config.pickle`, "
        "then place them in the same folder as this app."
    )
    st.stop()

seed_text = st.text_input("Enter the start of a sentence:", value="the sun was")

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Number of words to generate", min_value=1, max_value=15, value=3)
with col2:
    top_k = st.slider("Show top-K candidates for the next word", min_value=1, max_value=5, value=3)

if st.button("Predict", type="primary"):
    if not seed_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Predicting..."):
            generated_text, candidates = predict_next_words(
                seed_text, model, tokenizer, max_sequence_len, num_words=num_words, top_k=top_k
            )

        st.subheader("Generated text")
        st.success(generated_text)

        st.subheader(f"Top {top_k} candidates for the very next word")
        for word, prob in candidates:
            st.write(f"**{word}** — {prob:.2%}")
            st.progress(min(prob, 1.0))

st.divider()
st.caption(
    "This demo model was trained on a small sample corpus (`corpus.txt`) for "
    "illustration. For better predictions, retrain on a larger, more relevant "
    "text corpus using `train_model.py`."
)