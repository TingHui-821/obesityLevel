import streamlit as st
import numpy as np
import pandas as pd
import joblib
import keras
import altair as alt

# ----------------------------------------------------------------------
# Config / encodings
# ----------------------------------------------------------------------

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


LABEL_MAP = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III",
}

# Palette kept intuitive (blue -> green -> amber -> red) but tuned to sit
# nicely against the new light theme.
LABEL_COLORS = {
    "Insufficient_Weight": "#3B82F6",
    "Normal_Weight": "#10B981",
    "Overweight_Level_I": "#EAB308",
    "Overweight_Level_II": "#F59E0B",
    "Obesity_Type_I": "#F97316",
    "Obesity_Type_II": "#EF4444",
    "Obesity_Type_III": "#B91C1C",
}

MODEL_COLORS = {
    "Random Forest": "#6B8F5E",   # matcha green
    "SVM": "#E8829A",             # strawberry pink
    "KNN": "#C9A24B",             # toasted matcha-latte gold
    "ANN": "#9B6B8C",             # muted berry purple
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

    Bundled into a single joblib file: {"scaler": ..., "columns": ...}
    so the fitted scaler and its expected column order always travel
    together and can't drift out of sync with each other.
    """
    bundle = joblib.load("obesity_scaler_bundle.joblib")
    return bundle["scaler"], bundle["columns"]


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
# Page config + theme
# ----------------------------------------------------------------------

st.set_page_config(page_title="Obesity Level Predictor", page_icon="⚖️", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, span, label, div {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, h4, h5, h6, .hero-title {
    font-family: 'Fraunces', serif !important;
}

/* ---- Overall app background (adapts to Light/Dark via Streamlit's theme vars) ---- */
.stApp {
    background: linear-gradient(180deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
}

/* ---- Hero banner (matcha -> strawberry swirl, stays vivid in both themes) ---- */
.hero-banner {
    background: linear-gradient(120deg, #6B8F5E 0%, #A8C29A 38%, #F1AFC0 72%, #E8829A 100%);
    padding: 2rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px -12px rgba(232, 130, 154, 0.4);
}
.hero-title {
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
    text-shadow: 0 1px 8px rgba(0,0,0,0.08);
}
.hero-caption {
    color: #FDF3F0;
    font-size: 0.98rem;
    max-width: 720px;
}
.page-pill {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    color: #FFFFFF;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}

/* ---- Sidebar (deep matcha, fixed brand color, reads fine in either theme) ---- */
section[data-testid="stSidebar"] {
    background-color: #3B5240;
}
section[data-testid="stSidebar"] * {
    color: #F3EDE3;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-family: 'Fraunces', serif !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: #C7D6C4 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(243, 237, 227, 0.2);
}
/* Radio nav items styled like a menu */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(243, 237, 227, 0.15);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.35rem;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(232, 130, 154, 0.28);
}

/* ---- Buttons (strawberry, fixed brand color) ---- */
.stButton > button {
    background: linear-gradient(120deg, #E8829A, #D46A82);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1rem;
    box-shadow: 0 6px 16px -6px rgba(232, 130, 154, 0.55);
}
.stButton > button:hover {
    background: linear-gradient(120deg, #D46A82, #BF5570);
    color: #FFFFFF;
}

/* ---- Containers / cards (bg + text adapt to theme automatically) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border-color: rgba(168, 194, 154, 0.45) !important;
    background-color: var(--secondary-background-color);
    color: var(--text-color);
}
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid rgba(232, 130, 154, 0.3);
    background-color: var(--secondary-background-color);
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--text-color);
}

/* ---- Metric ---- */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(232, 130, 154, 0.35);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.65;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #D46A82;
    opacity: 1;
}
div[data-baseweb="tab-highlight"] {
    background-color: #D46A82 !important;
}

/* ---- Dataframe / table corners ---- */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(168, 194, 154, 0.4);
}

hr {
    border-color: rgba(168, 194, 154, 0.4);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(pill: str, title: str, caption: str):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="page-pill">{pill}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Model comparison charts (accuracy, precision, recall, F1, ROC AUC)
# ----------------------------------------------------------------------

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

        base = alt.Chart(df).encode(
            x=alt.X("Model:N", sort=None, title=None),
            y=alt.Y(f"{metric}:Q", title="%", scale=alt.Scale(zero=False)),
        )

        bars = base.mark_bar(clip=False, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            color=alt.Color("Model:N", scale=color_scale, legend=None),
            tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=".2f")],
        )

        labels = base.mark_text(
            clip=False,
            align="center",
            baseline="bottom",
            dy=-8,
            fontSize=14,
            fontWeight="bold",
            color="#0F172A",
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


# ----------------------------------------------------------------------
# Page: Exploratory Data Analysis (only built when this page is selected)
# ----------------------------------------------------------------------

def render_eda_page():
    hero(
        "Explore",
        "🔍 Exploratory Data Analysis",
        "Dataset: 2,111 survey responses (2,087 after removing 24 duplicates) on eating "
        "habits and physical condition, labeled with one of 7 obesity levels (NObeyesdad). "
        "Height, Weight, and BMI are shown here for context only — they're excluded from "
        "the model features to avoid leaking the answer into the prediction.",
    )

    with st.expander("📊 Target class distribution", expanded=True):
        st.markdown(
            "Classes are fairly balanced, ranging from **12.8%** "
            "(Insufficient Weight) to **16.8%** (Obesity Type I) of the "
            "dataset — so no class-imbalance correction (e.g. SMOTE) was "
            "needed before training."
        )
        st.image("eda_assets/fig1_target_distribution.png", use_container_width=True)

    with st.expander("📦 Boxplot: numeric feature spread & outliers"):
        st.markdown(
            "Shows each numeric feature's raw spread before cleaning. "
            "**Age** and **NCP** were deliberately left *out* of IQR capping — "
            "Age's higher values are real adult ages, not errors, and NCP is "
            "a near-discrete \"meals per day\" value where IQR flags its "
            "natural clustering as false outliers. The other four "
            "(FCVC, CH2O, FAF, TUE) were IQR-capped."
        )
        st.image("eda_assets/fig2_boxplot.png", use_container_width=True)

    with st.expander("📈 Distribution shape & skewness"):
        st.markdown(
            "Histogram + skewness score for each numeric feature. **Weight** "
            "and **BMI** are right-skewed (a longer tail toward higher "
            "values), which lines up with the dataset having more severe "
            "obesity classes than a general population sample would."
        )
        st.image("eda_assets/fig5_histograms.png", use_container_width=True)

    with st.expander("🧩 Pairwise relationships between key features"):
        st.markdown(
            "Age, BMI, FCVC (vegetable intake), FAF (physical activity), and "
            "CH2O (water intake), colored by obesity level. **BMI vs. Age** "
            "shows the clearest separation — higher obesity classes cluster "
            "at higher BMI across most ages, as expected."
        )
        st.image("eda_assets/fig3_pairplot.png", use_container_width=True)

    with st.expander("🔗 Correlation between numeric features"):
        st.markdown(
            "As expected, **Weight and BMI are strongly correlated** "
            "(BMI is derived from Weight and Height), which is exactly why "
            "both are excluded from the model's input features. Lifestyle "
            "features (FCVC, FAF, CH2O, TUE) show only weak correlation with "
            "each other, meaning they contribute fairly independent signal."
        )
        st.image("eda_assets/fig4_correlation.png", use_container_width=True)

    with st.expander("🧬 All attributes vs. obesity level"):
        st.markdown(
            "Categorical features (Gender, family history, MTRANS, etc.) as "
            "grouped counts; FCVC, NCP, CH2O, FAF, and TUE as density curves "
            "instead of counts, since those five are continuous "
            "(FCVC alone has 810 distinct values across 2,111 rows) rather "
            "than the small set of discrete answers they might look like. "
            "**Family history of overweight** shows the starkest split: it's "
            "heavily skewed toward \"yes\" in every obesity class above "
            "Normal Weight."
        )
        st.image("eda_assets/fig6_attributes_by_obesity.png", use_container_width=True)

    with st.expander("📋 Categorical features: plain frequency counts"):
        st.markdown(
            "Same 8 categorical features as above, but without splitting by "
            "obesity level — just how common each answer is overall. Most "
            "are heavily one-sided: **SMOKE** and **SCC** (calorie "
            "monitoring) are both over 95% \"no\", and **MTRANS** is "
            "dominated by Public Transportation, which is worth keeping in "
            "mind since these low-frequency categories give the model very "
            "few examples to learn from."
        )
        st.image("eda_assets/fig8_categorical_univariate.png", use_container_width=True)

    with st.expander("🌡️ Average feature values by obesity level"):
        st.markdown(
            "Mean value of each numeric feature per obesity class, ordered "
            "by severity. **Weight and BMI climb steadily** from Insufficient "
            "Weight through Obesity Type III, while lifestyle features like "
            "FAF (physical activity) trend gently downward — consistent with "
            "the behavioral story the model is trying to learn."
        )
        st.image("eda_assets/fig7_mean_heatmap.png", use_container_width=True)


# ----------------------------------------------------------------------
# Page: Model comparison (only built when this page is selected)
# ----------------------------------------------------------------------

def render_comparison_page(comparison_df: pd.DataFrame):
    hero(
        "Benchmark",
        "📊 Model Comparison",
        "All metrics computed once on the held-out test set (not from any single "
        "prediction). This is the basis for choosing which model performs best overall.",
    )
    render_comparison_charts(comparison_df)


# ----------------------------------------------------------------------
# Page: Predict
# ----------------------------------------------------------------------

def render_predict_page(model_choice, rf_model, knn_model, svm_model, ann_model,
                         real_scaler, minmax_scaler, minmax_cols):
    hero(
        "Predict",
        "⚖️ Obesity Level Predictor",
        "Predicts obesity category from eating habits and physical condition, using 4 "
        "trained models (Random Forest, KNN, SVM, ANN).",
    )

    with st.expander("ℹ️ About this app's inputs"):
        st.markdown(
            "Numeric inputs (Age, FCVC, NCP, CH2O, FAF, TUE) are scaled using the "
            "**actual fitted MinMaxScaler** from training (`obesity_scaler.joblib`), "
            "not guessed ranges. Categorical/ordinal encodings in the `CONFIG` "
            "block match the mappings used in the cleaning notebook directly."
        )

    st.markdown("### Enter your information")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("#### 🧍 Personal")
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
        with st.container(border=True):
            st.markdown("#### 🍽️ Eating habits")
            favc = st.selectbox("Frequent consumption of high-caloric food?", ["yes", "no"])
            fcvc = st.slider("Vegetable consumption frequency (1=never, 3=always)", 1.0, 3.0, 2.0, 0.1)
            ncp = st.slider("Number of main meals per day", 1.0, 4.0, 3.0, 0.5)
            caec = st.selectbox("Eating food between meals", ["no", "Sometimes", "Frequently", "Always"])
            calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
            scc = st.selectbox("Do you monitor calorie intake?", ["no", "yes"])

    with col3:
        with st.container(border=True):
            st.markdown("#### 🏃 Lifestyle")
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
            if model_name == "Random Forest":
                return rf_model.predict_proba(X_values)[0]

            if model_name == "KNN":
                return knn_model.predict_proba(X_values)[0]

            if model_name == "SVM":
                X_std = real_scaler.transform(X_values)
                return svm_model.predict_proba(X_std)[0]

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

            bars = base.mark_bar(clip=False, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                color=alt.Color("Color:N", scale=None),
                tooltip=["Category", alt.Tooltip("Probability", format=".2%")]
            )

            labels = base.mark_text(
                clip=False,
                align="center",
                baseline="bottom",
                dy=-8,
                fontSize=13,
                fontWeight="bold",
                color="#0F172A",
            ).encode(
                text=alt.Text("Probability:Q", format=".1%"),
            )

            chart = (bars + labels).properties(
                height=350,
                padding={"top": 20, "left": 5, "right": 5, "bottom": 5},
            )

            st.altair_chart(chart, use_container_width=True)

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


# ----------------------------------------------------------------------
# Sidebar navigation — pages only render their content when selected
# ----------------------------------------------------------------------

st.sidebar.markdown("## ⚖️ Obesity Predictor")
st.sidebar.caption("Navigate between sections")

page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict", "📊 Model Comparison", "🔍 Exploratory Data Analysis"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.divider()

comparison_df = load_comparison()

if page == "🔮 Predict":
    st.sidebar.header("Model selection")
    model_choice = st.sidebar.radio(
        "Which model should make the prediction?",
        ["Random Forest", "SVM", "KNN", "ANN", "Compare all 4"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.subheader("Model performance (test set)")
    st.sidebar.dataframe(comparison_df.set_index("Model"), use_container_width=True)

    rf_model, knn_model, svm_model, ann_model = load_models()
    real_scaler = load_svm_scaler()               # SVM's own fitted StandardScaler (SVM input only)
    minmax_scaler, minmax_cols = load_scaler()     # actual training-time MinMaxScaler (numeric inputs)

    render_predict_page(
        model_choice, rf_model, knn_model, svm_model, ann_model,
        real_scaler, minmax_scaler, minmax_cols,
    )

elif page == "📊 Model Comparison":
    render_comparison_page(comparison_df)

else:
    render_eda_page()
