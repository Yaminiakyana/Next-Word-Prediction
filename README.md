# LSTM Next Word Predictor (Streamlit)

A simple app that predicts the next word (or several words) after a phrase you type, using an LSTM language model trained with Keras/TensorFlow.

## Files

| File | Purpose |
|---|---|
| `corpus.txt` | Sample training text (replace with your own for better/domain-specific predictions) |
| `train_model.py` | Trains the LSTM and saves the model + tokenizer |
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `next_word_model.keras` | Trained model (already generated for you) |
| `tokenizer.pickle` | Fitted tokenizer (already generated for you) |
| `model_config.pickle` | Small config the app needs (already generated for you) |

The model is already trained and included, so you can run the app immediately. If you want to retrain (e.g. on your own text), see "Retraining" below.

## 1. Run it locally

```bash
# from inside this folder
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`. Type a phrase like "the sun was" and hit Predict.

## 2. Retraining on your own text (optional)

Replace `corpus.txt` with your own text (one sentence/line per line works best), then:

```bash
python train_model.py corpus.txt
```

This regenerates `next_word_model.keras`, `tokenizer.pickle`, and `model_config.pickle`. Note: the included sample corpus is tiny and purely illustrative — for genuinely useful predictions you'll want at least a few thousand lines of text relevant to your domain, and more training epochs (edit `EPOCHS` in `train_model.py`).

## 3. Deploy it — Streamlit Community Cloud (free, easiest)

1. **Push this folder to a GitHub repo.**
   ```bash
   git init
   git add .
   git commit -m "LSTM next word predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   Make sure `next_word_model.keras`, `tokenizer.pickle`, and `model_config.pickle` are committed too (don't put them in `.gitignore`) — the app needs them at runtime.

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **"New app"**, select your repo/branch, and set the main file path to `app.py`.

4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch your app — you'll get a public URL like `https://your-app-name.streamlit.app`.

That's it — any time you push new commits to the repo, the deployed app auto-updates.

### Notes on Streamlit Cloud
- Free tier apps sleep after inactivity and wake on the next visit (a few seconds delay).
- This is pinned to `tensorflow==2.21.0` (which bundles Keras 3) and `numpy==2.4.6` to match a known-working setup. If deploying to a platform with tighter memory limits (e.g. Streamlit Cloud's free tier), you can switch to `tensorflow-cpu` instead of `tensorflow` for a smaller install.
- If your model file grows large (bigger corpus/architecture), GitHub has a 100MB file size limit — use [Git LFS](https://git-lfs.github.com/) if needed.

## 4. Alternative deployment options

- **Hugging Face Spaces**: create a Space with the "Streamlit" SDK, upload these same files — very similar workflow to Streamlit Cloud, sometimes better for larger models.
- **Render / Railway**: create a new Web Service pointing at this repo with start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Docker + any cloud VM**: containerize with a simple Dockerfile (`FROM python:3.11-slim`, copy files, `pip install -r requirements.txt`, `CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]`) and run on any host that exposes port 8501.

## How it works (brief)

1. `train_model.py` builds n-gram sequences from every line of the corpus (e.g. "the cat sat" → `[the, cat]`, `[the, cat, sat]`) so the model learns to predict the next token given a growing context window.
2. The model is `Embedding → LSTM → Dense(softmax over vocabulary)`.
3. At inference, `app.py` tokenizes your typed text, pads it to the same length used in training, and feeds it through the model. The softmax output gives a probability for every word in the vocabulary; the highest-probability word is the prediction, and you can also see the top-K candidates.