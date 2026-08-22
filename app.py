import streamlit as st
import numpy as np
import pandas as pd
import joblib
import keras

# ----------------------------------------------------------------------
# CONFIG — edit this block if your original preprocessing differs
# ----------------------------------------------------------------------

# NOTE: numeric scaling no longer uses guessed min/max values. The real
# fitted MinMaxScaler (obesity_scaler.joblib) is loaded further down and
# used directly, so this file no longer needs to know or guess the
# training-set ranges at all — see scale_numeric_inputs().

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


@st.cache_resource
def load_scaler():
    """The real MinMaxScaler fit on the training set during cleaning
    (see: scaler.fit_transform(X_train[num_cols]) in the cleaning
    notebook). Loading the actual fitted object instead of guessing
    min/max bounds means this app can never drift out of sync with
    what the models were actually trained on.
    """
    scaler = joblib.load("obesity_scaler.joblib")
    scaler_cols = joblib.load("obesity_scaler_columns.joblib")
    return scaler, scaler_cols


def scale_numeric_inputs(inputs: dict, scaler, scaler_cols) -> dict:
    """Scales the numeric fields using the real fitted scaler, in the
    exact column order it was fit with, instead of a per-field guessed
    (value - lo) / (hi - lo) formula."""
    raw_row = pd.DataFrame([[inputs[c] for c in scaler_cols]], columns=scaler_cols)
    scaled_row = scaler.transform(raw_row)[0]
    return dict(zip(scaler_cols, scaled_row))


@st.cache_resource
def load_svm_scaler():
    """The real StandardScaler the SVM notebook itself fit and saved
    (svm_scaler.pkl) — no need to guess or reverse-engineer this one,
    the training notebook exports it directly."""
    return joblib.load("svm_scaler.pkl")


def build_feature_vector(inputs: dict, scaler, scaler_cols) -> pd.DataFrame:
    scaled_numeric = scale_numeric_inputs(inputs, scaler, scaler_cols)
    row = {
        "Gender": ENCODINGS["Gender"][inputs["Gender"]],
        "Age": scaled_numeric["Age"],
        "family_history_with_overweight": ENCODINGS["family_history_with_overweight"][inputs["family_history"]],
        "FAVC": ENCODINGS["FAVC"][inputs["FAVC"]],
        "FCVC": scaled_numeric["FCVC"],
        "NCP": scaled_numeric["NCP"],
        "CAEC": ENCODINGS["CAEC"][inputs["CAEC"]],
        "SMOKE": ENCODINGS["SMOKE"][inputs["SMOKE"]],
        "CH2O": scaled_numeric["CH2O"],
        "SCC": ENCODINGS["SCC"][inputs["SCC"]],
        "FAF": scaled_numeric["FAF"],
        "TUE": scaled_numeric["TUE"],
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

with st.expander("ℹ️ About this app's inputs"):
    st.markdown(
        "Numeric inputs (Age, FCVC, NCP, CH2O, FAF, TUE) are scaled using the "
        "**actual fitted MinMaxScaler** from training (`obesity_scaler.joblib`), "
        "not guessed ranges. Categorical/ordinal encodings in the `CONFIG` "
        "block match the mappings used in the cleaning notebook directly."
    )

# ----------------------------------------------------------------------
# Model comparison charts (accuracy, precision, recall, F1, ROC AUC)
# ----------------------------------------------------------------------

import altair as alt

# Fixed color per model, kept consistent across every metric tab so the
# same model always shows the same color throughout the app.
MODEL_COLORS = {
    "Random Forest": "#EF4444",
    "SVM": "#3B82F6",
    "KNN": "#22C55E",
    "ANN": "#A855F7",
}


def render_comparison_charts(df: pd.DataFrame):
    """Bar charts comparing all models across every metric in model_comparison.csv.

    Uses Altair (bundled with Streamlit, no extra dependency) instead of
    st.bar_chart so each model's bar can have its own distinct, consistent
    color — st.bar_chart only colors by column/series, not by category,
    so it can't do per-model colors on a single-metric chart.

    Expects df with a 'Model' column plus one or more numeric metric
    columns (e.g. Accuracy, Precision_Macro, Recall_Macro, F1_Macro,
    ROC_AUC_Macro), values already in percent (0-100).
    """
    metric_cols = [c for c in df.columns if c != "Model"]
    models_present = df["Model"].tolist()
    color_scale = alt.Scale(
        domain=models_present,
        range=[MODEL_COLORS.get(m, "#94A3B8") for m in models_present],
    )

    def bar_chart_for(metric: str):
        # zero=False lets Vega-Lite auto-zoom the y-axis to the data's
        # natural range instead of forcing it to start at 0 — otherwise
        # metrics clustered close together (e.g. 86%, 87%, 86%, 98%)
        # produce bars that look nearly identical across tabs even though
        # the underlying values differ.
        #
        # NOTE: an earlier version of this also set an explicit `domain`
        # on the scale to force extra headroom above the tallest bar for
        # the value label. That rendered fine in offline testing but
        # blanked the *actual* chart in this app's Streamlit/Vega-Lite
        # runtime (bars and all) — so it's deliberately avoided now.
        # zero=False alone is the same config the chart used back when
        # bars were rendering correctly; headroom for the label is added
        # below via chart-level padding + clip=False instead, which
        # doesn't touch the data scale at all.
        base = alt.Chart(df).encode(
            x=alt.X("Model:N", sort=None, title=None),
            y=alt.Y(f"{metric}:Q", title="%", scale=alt.Scale(zero=False)),
        )

        bars = base.mark_bar(clip=False).encode(
            color=alt.Color("Model:N", scale=color_scale, legend=None),
            tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".2f")],
        )

        # Value label above each bar's top edge. clip=False lets it draw
        # into the padding area above the plot instead of being cut off
        # by the axis boundary. Single mark with both a dark fill and a
        # light stroke so it stays readable on light or dark themes.
        labels = base.mark_text(
            clip=False,
            align="center",
            baseline="bottom",
            dy=-8,
            fontSize=20,
            fontWeight="bold",
            color="#000000",
        ).encode(
            text=alt.Text(f"{metric}:Q", format=".2f"),
        )

        return (bars + labels).properties(
            height=320,
            padding={"top": 25, "left": 5, "right": 5, "bottom": 5},
        )

    tabs = st.tabs([m.replace("_", " ") for m in metric_cols])
    for tab, metric in zip(tabs, metric_cols):
        with tab:
            st.caption(
                f"Range shown: {df[metric].min():.2f}% – {df[metric].max():.2f}% "
                "(zoomed in — not 0-100 — so small differences between models are visible)"
            )
            st.altair_chart(
                bar_chart_for(metric),
                use_container_width=True,
                key=f"cmp_chart_{metric}",
            )

    with st.expander("Show all metrics side-by-side (grouped)"):
        melted = df.melt(id_vars="Model", value_vars=metric_cols,
                          var_name="Metric", value_name="Value")
        # NOTE: `column`/`row` facet encodings must be applied to the
        # already-layered chart, not baked into a shared base that gets
        # layered — putting it in the base and then doing bars + labels
        # produces an invalid spec (SchemaValidationError). Layer first,
        # facet second.
        base = alt.Chart(melted).encode(
            x=alt.X("Model:N", sort=None, title=None),
            y=alt.Y("Value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
        )
        bars = base.mark_bar().encode(
            color=alt.Color("Model:N", scale=color_scale, legend=alt.Legend(title="Model")),
            tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".2f")],
        )
        labels = base.mark_text(
            align="center", baseline="bottom", dy=-2, fontSize=9,
        ).encode(
            text=alt.Text("Value:Q", format=".1f"),
        )
        layered = alt.layer(bars, labels).properties(height=280, width=120)
        grouped = layered.facet(column=alt.Column("Metric:N", title=None))
        st.altair_chart(grouped, use_container_width=False, key="cmp_chart_grouped")


st.subheader("📊 Model comparison (test set)")
st.caption(
    "All metrics computed once on the held-out test set (not from any "
    "single prediction below). This is the basis for choosing which "
    "model performs best overall."
)
render_comparison_charts(load_comparison())

st.divider()


rf_model, knn_model, svm_model, ann_model = load_models()
comparison_df = load_comparison()
real_scaler = load_svm_scaler()                    # SVM's own fitted StandardScaler (SVM input only)
minmax_scaler, minmax_cols = load_scaler()         # actual training-time MinMaxScaler (numeric inputs)

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
        "Compare it to the estimate below — they answer different "
        "questions: this is 'what BMI category are you in right now', "
        "the model below is 'what obesity category does your current "
        "lifestyle pattern-match to' (both describe your PRESENT state, "
        "not a future forecast)."
    )
    st.divider()

    X = build_feature_vector(inputs, minmax_scaler, minmax_cols)
    X_values = X.values.astype(float)

    def predict_proba(model_name):
        # RF: verified (via its training notebook) — trained directly on the
        # raw min-max/ordinal feature vector, no extra scaling. Keep as-is.
        if model_name == "Random Forest":
            return rf_model.predict_proba(X_values)[0]
        # KNN: standardizes internally via its own embedded pipeline —
        # keep feeding it the raw vector, do not double-scale.
        if model_name == "KNN":
            return knn_model.predict_proba(X_values)[0]
        # SVM: verified (via its training notebook) — trained on features
        # standardized with SVM's own StandardScaler. Apply it before predict.
        if model_name == "SVM":
            X_std = real_scaler.transform(X_values)
            return svm_model.predict_proba(X_std)[0]
        # ANN: verified (via its training notebook) — trained directly on
        # the raw min-max/ordinal feature vector, same as RF. No scaling.
        if model_name == "ANN":
            return ann_model.predict(X_values, verbose=0)[0]
        raise ValueError(model_name)

    def show_result(model_name, proba):
        # Indented correctly inside the function
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

        # 1. Create the DataFrame in the specific order of LABEL_MAP
        categories = [LABEL_MAP[i].replace("_", " ") for i in range(len(proba))]
        colors = [LABEL_COLORS[LABEL_MAP[i]] for i in range(len(proba))]
        
        proba_df = pd.DataFrame({
            "Category": categories,
            "Probability": proba,
            "Color": colors
        })

        # 2. Use Altair for the chart to force the categorical order and use custom colors
        base = (
            alt.Chart(proba_df)
            .encode(
                x=alt.X("Category:N", sort=categories, title="Obesity Level"),
                y=alt.Y("Probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            )
        )

        bars = base.mark_bar(clip=False).encode(
            color=alt.Color("Color:N", scale=None),
            tooltip=["Category", alt.Tooltip("Probability", format=".2%")]
        )

        # Value label above each bar, shown as a percentage. clip=False
        # lets labels on the tallest bar draw past the plot edge instead
        # of being cut off (kept as a single mark with fill+stroke, and
        # scale left untouched — see the comment in bar_chart_for above
        # for why a data-scale domain override is avoided here).
        labels = base.mark_text(
            clip=False,
            align="center",
            baseline="bottom",
            dy=-8,
            fontSize=20,
            fontWeight="bold",
            color="#000000",
        ).encode(
            text=alt.Text("Probability:Q", format=".1%"),
        )

        chart = (bars + labels).properties(
            height=350,
            padding={"top": 20, "left": 5, "right": 5, "bottom": 5},
        )
        
        st.altair_chart(chart, use_container_width=True)

    # These lines are now back at the 'if st.button' level
    st.subheader("🧠 Predicted current obesity category (from your habits)")
    st.caption(
        "This is the model's estimate of which obesity category best "
        "matches your CURRENT lifestyle habits — it is not a future "
        "forecast or a risk score. It can differ from your BMI category "
        "above because it's inferred indirectly from behaviour patterns, "
        "not calculated from your actual height/weight."
    )

    if model_choice == "Compare all 4":
        tabs = st.tabs(["Random Forest", "SVM", "KNN", "ANN"])
        for tab, name in zip(tabs, ["Random Forest", "SVM", "KNN", "ANN"]):
            with tab:
                show_result(name, predict_proba(name))
    else:
        show_result(model_choice, predict_proba(model_choice))

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(X, use_container_width=True)
