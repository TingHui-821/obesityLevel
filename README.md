# Obesity Level Predictor (Streamlit)

Predicts obesity category from lifestyle/eating-habit inputs using 4 trained
models: Random Forest, KNN, SVM, and an ANN (Keras).

## Files in this repo
- `app.py` — the Streamlit app
- `requirements.txt` — pinned dependencies
- `obesity_rf_model.joblib`, `obesity_knn_model.joblib`, `obesity_svm_model.joblib` — sklearn models
- `obesity_ann_model.keras` — Keras model
- `model_comparison.csv` — accuracy/precision/recall/F1/ROC-AUC per model

## ⚠️ Before deploying
This app reconstructs the scaling/encoding used during training because no
`scaler.pkl`/`encoder.pkl` was saved alongside the models. The assumptions
are documented at the top of `app.py` (`CONFIG` block). Double-check
`RAW_MIN_MAX`, `ENCODINGS`, and `LABEL_MAP` against your original training
notebook — if they don't match, predictions will run without errors but be
wrong.

## Deploy on Streamlit Community Cloud
1. Create a new **public** GitHub repo and push everything in this folder
   to it (see the step-by-step in chat for doing this from Google Colab).
2. Go to https://share.streamlit.io → **New app**.
3. Pick your repo/branch, set **Main file path** to `app.py`, click **Deploy**.
4. First boot takes a few minutes (installing TensorFlow). Subsequent
   reboots are fast.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
