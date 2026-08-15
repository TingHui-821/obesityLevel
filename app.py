"""
Obesity Level Prediction App
-----------------------------
Streamlit app that loads 4 pre-trained models (Random Forest, KNN, SVM, ANN)
and predicts an obesity category from lifestyle / physical inputs.

IMPORTANT — read this before trusting the predictions:
The training CSVs you provided (Cleaned_Obesity_Train.csv / Cleaned_Obesity_Test.csv)
were ALREADY preprocessed: numeric columns are min-max scaled to [0, 1] and
categorical columns are already label/one-hot encoded. No scaler.pkl or
encoder.pkl was uploaded alongside the models, so this app has to guess how
raw human-readable inputs (e.g. "Age = 25", "Gender = Male") map back to
those encoded values.

The mappings below (RAW_MIN_MAX, ENCODINGS, LABEL_MAP) are my best
reconstruction based on the classic UCI/Kaggle "Obesity Levels" dataset,
which this data clearly derives from. If your original notebook used
different encoding choices (e.g. a different LabelEncoder class order, or
different scaler bounds), predictions will be WRONG even though the app
runs fine. Everything you'd need to change lives in the CONFIG block below —
open your training notebook, compare it against these dictionaries, and
edit them to match if needed.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import keras

# ----------------------------------------------------------------------
# CONFIG — edit this block if your original preprocessing differs
# ----------------------------------------------------------------------

# Min/max used to reproduce the training set's min-max scaling.
# These are the standard ranges for the UCI Obesity dataset this data
# is derived from. Adjust if your notebook used different bounds.
RAW_MIN_MAX = {
    "Age": (14, 61),
    "FCVC": (1, 3),   # vegetable consumption frequency
    "NCP": (1, 4),    # number of main meals
    "CH2O": (1, 3),   # daily water intake (liters)
    "FAF": (0, 3),    # physical activity frequency
    "TUE": (0, 2),    # time using technology devices
}

# Binary / ordinal encodings assumed during training
ENCODINGS = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},           # frequent high-caloric food
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},            # monitors calories
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},   # eating between meals
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},   # alcohol consumption
}

MTRANS_OPTIONS = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
# "Automobile" was dropped as the one-hot baseline (all MTRANS_* columns = 0)

# NOTE: Height and Weight are deliberately NOT in FEATURE_ORDER and never
# reach the trained models. NObeyesdad (the target) is essentially a
# bucketed version of BMI = Weight / Height^2, so feeding Height/Weight (or
# BMI) into the model would leak the answer straight to it and inflate
# accuracy without the model learning anything about behaviour. They're
# collected below ONLY to show a simple, direct BMI calculation next to
# the model's behavioural prediction — two independent numbers, not one
# feeding the other.
def bmi_to_label(bmi: float) -> str:
    if bmi < 18.5: return "Insufficient_Weight"
    elif bmi < 25: return "Normal_Weight"
    elif bmi < 27.5: return "Overweight_Level_I"
    elif bmi < 30: return "Overweight_Level_II"
    elif bmi < 35: return "Obesity_Type_I"
    elif bmi < 40: return "Obesity_Type_II"
    else: return "Obesity_Type_III"

FEATURE_ORDER = [
    "Gender", "Age", "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC",
    "MTRANS_Bike", "MTRANS_Motorbike", "MTRANS_Public_Transportation", "MTRANS_Walking",
]

# Label order — MUST match the target_order used when NObeyesdad was
# encoded during training (from data_cleaning.py: severity order, NOT
# alphabetical). Using the wrong order here doesn't crash anything, it
# just silently mislabels the model's output (e.g. showing "Obesity Type
# III" when the model actually predicted "Obesity Type I"), which is what
# was causing the wildly inconsistent-looking predictions across models.
LABEL_MAP = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

LABEL_COLORS = {
    "Insufficient_Weight": "#3B82F6",
    "Normal_Weight": "#22C55E",
    "Overweight_Level_I": "#EAB308",
    "Overweight_Level_II": "#F59E0B",
    "Obesity_Type_I": "#F97316",
    "Obesity_Type_II": "#EF4444",
    "Obesity_Type_III": "#B91C1C",
}

# ----------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------

@st.cache_resource
def load_models():
    rf = joblib.load("obesity_rf_model.joblib")
    knn = joblib.load("obesity_knn_model.joblib")
    svm = joblib.load("obesity_svm_model.joblib")
    ann = keras.models.load_model("obesity_ann_model.keras")
    return rf, knn, svm, ann


@st.cache_data
def load_comparison():
    return pd.read_csv("model_comparison.csv")


def scale(value, key):
    lo, hi = RAW_MIN_MAX[key]
    return (value - lo) / (hi - lo)


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    row = {
        "Gender": ENCODINGS["Gender"][inputs["Gender"]],
        "Age": scale(inputs["Age"], "Age"),
        "family_history_with_overweight": ENCODINGS["family_history_with_overweight"][inputs["family_history"]],
        "FAVC": ENCODINGS["FAVC"][inputs["FAVC"]],
        "FCVC": scale(inputs["FCVC"], "FCVC"),
        "NCP": scale(inputs["NCP"], "NCP"),
        "CAEC": ENCODINGS["CAEC"][inputs["CAEC"]],
        "SMOKE": ENCODINGS["SMOKE"][inputs["SMOKE"]],
        "CH2O": scale(inputs["CH2O"], "CH2O"),
        "SCC": ENCODINGS["SCC"][inputs["SCC"]],
        "FAF": scale(inputs["FAF"], "FAF"),
        "TUE": scale(inputs["TUE"], "TUE"),
        "CALC": ENCODINGS["CALC"][inputs["CALC"]],
        "MTRANS_Bike": inputs["MTRANS"] == "Bike",
        "MTRANS_Motorbike": inputs["MTRANS"] == "Motorbike",
        "MTRANS_Public_Transportation": inputs["MTRANS"] == "Public_Transportation",
        "MTRANS_Walking": inputs["MTRANS"] == "Walking",
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(page_title="Obesity Level Predictor", page_icon="⚖️", layout="wide")

st.title("⚖️ Obesity Level Predictor")
st.caption(
    "Predicts obesity category from eating habits and physical condition, "
    "using 4 trained models (Random Forest, KNN, SVM, ANN)."
)

with st.expander("⚠️ About the accuracy of this app's inputs (read once)"):
    st.markdown(
        "The uploaded training data was already scaled/encoded, and no scaler "
        "or encoder file was provided with the models. This app reconstructs "
        "that preprocessing using standard assumptions for the UCI Obesity "
        "dataset. If your notebook encoded things differently, edit the "
        "`CONFIG` block at the top of `app.py` to match."
    )

rf_model, knn_model, svm_model, ann_model = load_models()
comparison_df = load_comparison()

st.sidebar.header("Model selection")
model_choice = st.sidebar.radio(
    "Which model should make the prediction?",
    ["Random Forest", "SVM", "KNN", "ANN", "Compare all 4"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Model performance (test set)")
st.sidebar.dataframe(comparison_df.set_index("Model"), use_container_width=True)

st.header("Enter your information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 14, 61, 25)
    family_history = st.selectbox("Family history of overweight?", ["yes", "no"])
    smoke = st.selectbox("Do you smoke?", ["no", "yes"])
    st.caption(
        "Height & weight (optional) — used only to show your BMI for "
        "reference. They are **not** sent to the prediction model."
    )
    height_m = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.5)

with col2:
    st.subheader("Eating habits")
    favc = st.selectbox("Frequent consumption of high-caloric food?", ["yes", "no"])
    fcvc = st.slider("Vegetable consumption frequency (1=never, 3=always)", 1.0, 3.0, 2.0, 0.1)
    ncp = st.slider("Number of main meals per day", 1.0, 4.0, 3.0, 0.5)
    caec = st.selectbox("Eating food between meals", ["no", "Sometimes", "Frequently", "Always"])
    calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
    scc = st.selectbox("Do you monitor calorie intake?", ["no", "yes"])

with col3:
    st.subheader("Lifestyle")
    ch2o = st.slider("Daily water intake (liters)", 1.0, 3.0, 2.0, 0.1)
    faf = st.slider("Physical activity frequency (days/week, 0-3)", 0.0, 3.0, 1.0, 0.5)
    tue = st.slider("Time using technology devices (0=low, 2=high)", 0.0, 2.0, 1.0, 0.1)
    mtrans = st.selectbox("Main mode of transportation", MTRANS_OPTIONS)

inputs = {
    "Gender": gender,
    "Age": age,
    "family_history": family_history,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "CALC": calc,
    "MTRANS": mtrans,
}

st.divider()

if st.button("🔍 Predict obesity level", type="primary", use_container_width=True):
    # --- BMI reference (independent of the model, calculated directly) ---
    bmi = weight_kg / (height_m ** 2)
    bmi_label = bmi_to_label(bmi)
    bmi_color = LABEL_COLORS.get(bmi_label, "#666")

    st.subheader("📏 Your BMI (direct calculation)")
    bcol1, bcol2 = st.columns([1, 2])
    with bcol1:
        st.metric("BMI", f"{bmi:.1f}")
    with bcol2:
        st.markdown(
            f"Category: <span style='color:{bmi_color}; font-weight:600'>"
            f"{bmi_label.replace('_', ' ')}</span>",
            unsafe_allow_html=True,
        )
    st.caption(
        "This is a direct BMI formula result, not a model prediction. "
        "Compare it to the behavioural prediction below — they answer "
        "different questions: this is 'what BMI category are you in right "
        "now', the model below is 'what does your lifestyle suggest'."
    )
    st.divider()

    X = build_feature_vector(inputs)
    X_values = X.values.astype(float)

    def predict_proba(model_name):
        if model_name == "Random Forest":
            return rf_model.predict_proba(X_values)[0]
        if model_name == "SVM":
            return svm_model.predict_proba(X_values)[0]
        if model_name == "KNN":
            return knn_model.predict_proba(X_values)[0]
        if model_name == "ANN":
            return ann_model.predict(X_values, verbose=0)[0]
        raise ValueError(model_name)

    def show_result(model_name, proba):
        pred_idx = int(np.argmax(proba))
        pred_label = LABEL_MAP[pred_idx]
        confidence = float(proba[pred_idx]) * 100
        color = LABEL_COLORS.get(pred_label, "#666")

        st.markdown(
            f"### {model_name} prediction: "
            f"<span style='color:{color}'>{pred_label.replace('_',' ')}</span> "
            f"({confidence:.1f}% confidence)",
            unsafe_allow_html=True,
        )
        proba_df = pd.DataFrame({
            "Category": [LABEL_MAP[i].replace("_", " ") for i in range(len(proba))],
            "Probability": proba,
        }).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Category"))

    st.subheader("🧠 Behavioural risk prediction (from your lifestyle habits)")

    if model_choice == "Compare all 4":
        tabs = st.tabs(["Random Forest", "SVM", "KNN", "ANN"])
        for tab, name in zip(tabs, ["Random Forest", "SVM", "KNN", "ANN"]):
            with tab:
                show_result(name, predict_proba(name))
    else:
        show_result(model_choice, predict_proba(model_choice))

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(X, use_container_width=True)
