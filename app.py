# app.py
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# ---------------------------
# Page config & CSS
# ---------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor 🚢",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; color: #e6eef8; }
h1 { color: #00BFFF; text-align: center; font-weight:800; }
.help-text { color:#9aa3ad; font-size:13px; margin-top:-8px; margin-bottom:12px; }
.stButton>button { background-color:#00BFFF; color: white; border-radius:10px; padding:8px 20px; font-size:16px; }
.stButton>button:hover { background-color:#009ACD; transform:scale(1.03); }
.prediction { text-align:center; font-weight:700; font-size:20px; color:#FFD700; margin-top:12px; }
.footer { color:#9aa3ad; font-size:12px; text-align:center; padding-top:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"⚠️ Failed to load `{MODEL_PATH}`. Ensure model.pkl is in the same folder.")
    st.exception(e)
    st.stop()

# ---------------------------
# Header
# ---------------------------
st.title("🚢 Titanic Survival Predictor")
st.markdown(
    "<p style='text-align:center;'>Answer a few simple questions to get a survival probability estimate.<br>"
    "<i>(This is a statistical model trained on historical Titanic data.)</i></p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Try sample passenger profiles")
sample = st.sidebar.selectbox(
    "Choose a scenario (or use custom inputs below)",
    (
        "Custom - fill details below",
        "1st Class Female (High survival)",
        "3rd Class Male (Low survival)",
        "Child with guardian (Higher survival)",
        "Single Middle-class Male (Typical case)"
    )
)

st.sidebar.markdown("### Model & environment")
st.sidebar.text(f"scikit-learn: {sklearn.__version__}")
st.sidebar.markdown("⚠️ Model may have been trained using a different sklearn version.")

# ---------------------------
# Default values for scenarios
# ---------------------------
def scenario_defaults(name):
    if name == "1st Class Female (High survival)":
        return dict(pclass=1, sex="Female", age=28, family=0, fare=80.0, embarked="Cherbourg (C)")
    if name == "3rd Class Male (Low survival)":
        return dict(pclass=3, sex="Male", age=35, family=2, fare=7.25, embarked="Southampton (S)")
    if name == "Child with guardian (Higher survival)":
        return dict(pclass=2, sex="Male", age=8, family=2, fare=21.0, embarked="Southampton (S)")
    if name == "Single Middle-class Male (Typical case)":
        return dict(pclass=2, sex="Male", age=30, family=0, fare=13.0, embarked="Southampton (S)")
    return dict(pclass=3, sex="Male", age=25, family=0, fare=32.0, embarked="Southampton (S)")

defaults = scenario_defaults(sample)

# ---------------------------
# Input form
# ---------------------------
col1, col2 = st.columns([1, 1])

with col1:
    pclass = st.selectbox(
        "🎟️ Ticket class", [1, 2, 3],
        index=[1, 2, 3].index(defaults['pclass']),
        help="1 = First (luxury), 2 = Second, 3 = Third (economy)."
    )
    st.markdown("<div class='help-text'>Higher classes generally had higher survival rates.</div>",
                unsafe_allow_html=True)

    age = st.slider(
        "🎂 Age (years)", 0, 100, int(defaults['age']),
        help="Age in years. Children often had higher priority during evacuation."
    )
    st.markdown("<div class='help-text'>Younger passengers tended to have better survival rates.</div>",
                unsafe_allow_html=True)

    family = st.number_input(
        "👪 Family members travelling with passenger", 0, 10, int(defaults['family']),
        help="Total family onboard (siblings/spouses/parents/children)."
    )
    st.markdown("<div class='help-text'>Family could help, but very large families sometimes decreased survival odds.</div>",
                unsafe_allow_html=True)

with col2:
    sex = st.radio(
        "⚧️ Gender", ("Male", "Female"),
        index=(0 if defaults['sex'] == "Male" else 1),
        help="Women had significantly higher survival probability historically."
    )
    st.markdown("<div class='help-text'>Women had higher survival probability.</div>",
                unsafe_allow_html=True)

    fare = st.number_input(
        "💰 Ticket fare (USD)", 0.0, 1000.0, float(defaults['fare']),
        format="%.2f", help="Higher fares often meant better cabins and lifeboat access."
    )
    st.markdown("<div class='help-text'>Higher fares often correlated with higher survival.</div>",
                unsafe_allow_html=True)

    embarked = st.selectbox(
        "🛳️ Boarding port",
        ("Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"),
        index=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"].index(defaults['embarked'])
    )
    st.markdown("<div class='help-text'>Boarding location influenced passenger demographics.</div>",
                unsafe_allow_html=True)

# ---------------------------
# Prepare model input
# ---------------------------
sex_male = 1 if sex == "Male" else 0
emb_code = embarked[0]  # S, C, Q
emb_C = 1 if emb_code == "C" else 0
emb_Q = 1 if emb_code == "Q" else 0

X_input = np.array([[pclass, age, family, 0, fare, sex_male, emb_C, emb_Q]])

# ---------------------------
# Summary
# ---------------------------
st.markdown("### 🧾 Passenger summary")
summary_df = pd.DataFrame({
    "Feature": ["Ticket class", "Gender", "Age", "Family members", "Fare (USD)", "Boarded at"],
    "Value": [pclass, sex, age, family, f"${fare:.2f}", embarked]
})
st.table(summary_df)
st.markdown("---")

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict survival chance"):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0][1]
        else:
            score = model.decision_function(X_input)[0]
            proba = 1 / (1 + np.exp(-score))

        perc = proba * 100
        st.markdown(f"### 🎯 Survival probability: **{perc:.2f}%**")
        st.progress(min(max(int(perc), 0), 100))

        if perc >= 80:
            st.success("💪 Very high chance of survival.")
        elif perc >= 60:
            st.success("🙂 High chance of survival.")
        elif perc >= 40:
            st.info("😐 Moderate chance of survival.")
        elif perc >= 20:
            st.warning("😟 Low chance of survival.")
        else:
            st.error("⚠️ Very low chance of survival.")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


else:
    st.markdown(
        "<div class='footer'>Tip: Select a sample scenario or fill inputs above and click Predict.</div>",
        unsafe_allow_html=True
    )# ---------------------------
# 🎯 Super Simple Feature Importance for Common Users
# ---------------------------
st.markdown("---")
st.markdown("### 🔍 Top 3 Factors Affecting Survival (Simple Explanation)")

try:
    # Extract coefficients from model
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
    else:
        st.info("Feature importance unavailable for this model.")
        coefs = None

    if coefs is not None:
        feature_names = ['Ticket Class','Age','Family Size','Parch','Fare','Male','Embarked C','Embarked Q']

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Weight": coefs
        })

        # Get absolute impact values
        df_imp["Impact"] = df_imp["Weight"].abs()

        # Pick top 3 strongest effects
        top3 = df_imp.sort_values(by="Impact", ascending=False).head(3)

        # For display: convert technical names → simple English
        rename_map = {
            "Male": "Gender (Being male reduces survival)",
            "Ticket Class": "Ticket Class (Higher class = safer)",
            "Fare": "Ticket Fare (Higher fare = better cabins)",
            "Age": "Age (Children favored more)",
            "Family Size": "Family With You",
            "Embarked C": "Boarding Port: Cherbourg",
            "Embarked Q": "Boarding Port: Queenstown",
            "Parch": "Parents/Children with you"
        }
        top3["Readable"] = top3["Feature"].map(rename_map)

        # Simple bar chart
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.barh(top3["Readable"], top3["Impact"], color="#4FC3F7")
        ax.set_xlabel("Impact Strength (Higher = more influence)")
        ax.set_ylabel("")
        ax.set_title("🔝 Top 3 Most Important Factors", pad=15)
        
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

        st.pyplot(fig)

        # Human explanation
        st.markdown("### 🧠 Simple Explanation")
        for _, row in top3.iterrows():
            if row["Feature"] == "Male":
                st.write("🧑 **Gender** plays the biggest role — historically women were prioritized for lifeboats.")
            elif row["Feature"] == "Ticket Class":
                st.write("🎟 **Ticket Class** mattered — 1st class passengers had better access to lifeboats.")
            elif row["Feature"] == "Fare":
                st.write("💰 **Fare** indicates cabin quality — richer passengers had better survival chances.")
            elif row["Feature"] == "Age":
                st.write("🎂 **Age** mattered — children were rescued first.")
            elif row["Feature"] == "Family Size":
                st.write("👪 **Traveling with family** had mixed impact — could help or delay.")
            elif row["Feature"] == "Embarked C":
                st.write("🛳 **Boarding at Cherbourg** correlated with higher survival.")
            elif row["Feature"] == "Embarked Q":
                st.write("🛳 **Boarding at Queenstown** had lower survival rates.")
            elif row["Feature"] == "Parch":
                st.write("👨‍👩‍👧 **Traveling with parents/children** affected survival.")

except Exception as e:
    st.warning("Unable to compute simplified feature impacts.")
    st.exception(e)

