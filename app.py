import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import joblib
import pandas as pd
import numpy as np

# Sklearn tools used in income prediction and heart disease prediction preprocessing pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Hugging Face Hub
from huggingface_hub import hf_hub_download


# =========================
# Streamlit Page Configuration
# =========================
st.set_page_config(
    page_title="ML Inference Hub", 
    layout="wide",
    page_icon="🤖"
)


# =========================
# Session State Initialization
# =========================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Overview"


# =========================
# Custom Styling for Streamlit
# =========================
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1518932945647-7a1c969f8be2?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", 
        caption="ML Hub v1.0"
    )
    st.title("🔍 Navigation")

    # Sidebar radio selection
    page = st.radio(
        "Select a project:", 
        ["🏠 Overview", "🎬 Sentiment Analysis", "💼 Income Prediction", "🫀 Heart Disease Prediction"],
        index=["🏠 Overview", "🎬 Sentiment Analysis", "💼 Income Prediction", "🫀 Heart Disease Prediction"].index(st.session_state.page)
    )
    # Update session state when sidebar changes
    st.session_state.page = page
    st.markdown("---")
    st.caption("Powered by Streamlit & Transformers")


# =========================
# CURRENT PAGE
# =========================
page = st.session_state.page


# =========================
# Helper function for one-click Overview navigation
# =========================
def go_to(page_name):
    st.session_state.page = page_name


# =========================
# PAGE 1: Overview
# =========================
if page == "🏠 Overview":
    st.title("Welcome to the AI & ML Inference Hub 🤖")
    st.image(
        "https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", 
        use_container_width=True
    )
    st.markdown("### Select a project to explore and interact with")
    st.write("This platform brings together advanced AI and ML projects for interactive testing and exploration.")

    # Project cards
    col1, col2, col3 = st.columns(3)

    # 🎬 Sentiment Analysis
    with col1:
        st.button("🎬 Sentiment Analysis", on_click=go_to, args=("🎬 Sentiment Analysis",))
        st.subheader("🎬 Sentiment Analysis")
        st.write("Analyze IMDB movie reviews to detect positive or negative sentiment using DistilBERT.")

    # 💼 Income Prediction
    with col2:
        st.button("💼 Income Prediction", on_click=go_to, args=("💼 Income Prediction",))
        st.subheader("💼 Income Prediction")
        st.write("Predict if an individual earns >$50K based on census demographic data.")

    # 🫀 Heart Disease
    with col3:
        st.button("🫀 Heart Disease", on_click=go_to, args=("🫀 Heart Disease Prediction",))
        st.subheader("🫀 Heart Disease")
        st.write("Assess medical risk factors to predict the likelihood of heart disease.")


# =========================
# PAGE 2: Sentiment Analysis
# =========================
elif page == "🎬 Sentiment Analysis":
    st.title("🎬 Sentiment Analysis")
    st.markdown("### IMDB Movie Review Classifier")

    col_main, col_img = st.columns([2,1])
    with col_img:
        st.image(
            "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", 
            use_container_width=True
        )

    with col_main:
        # Load DistilBERT model from Hugging Face Hub
        @st.cache_resource
        def load_model():
            """Load fine-tuned DistilBERT model from HF Hub."""
            repo_id = "nnejere/fine_tuned_distilbert_imdb"
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
            model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(tokenizer_path))
            model = AutoModelForSequenceClassification.from_pretrained(os.path.dirname(model_path))
            return tokenizer, model
        
        tokenizer, model = load_model()

        # Text Input
        text = st.text_area(
            "📝 Enter a movie review below:", 
            height=150, 
            placeholder="This movie was fantastic! The plot was intriguing..."
        )

        # Prediction
        if st.button("Analyze Sentiment"):
            try:
                if text.strip():
                    with st.spinner("Analyzing text patterns..."):
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            preds = torch.argmax(outputs.logits, dim=1).item()
                        sentiment = "😊 Positive" if preds == 1 else "😞 Negative"
                        st.markdown("---")
                        st.subheader("Analysis Result")
                        if preds == 1:
                            st.success(f"**Predicted Sentiment:** {sentiment}")
                            st.balloons()
                        else:
                            st.error(f"**Predicted Sentiment:** {sentiment}")
                else:
                    st.warning("Please enter a review to analyze.")
            except Exception as e:
                st.error(f"An error occurred during sentiment analysis: {e}")


# =========================
# PAGE 3: Income Prediction
# =========================
elif page == "💼 Income Prediction":
    st.title("💼 Income Classification")
    st.markdown("Predict whether an individual earns **above or below $50K/year**.")

    # Custom preprocessing functions
    def log_transform(X):
        X = X.copy()
        X["capital_gain"] = np.log1p(X["capital_gain"])
        X["capital_loss"] = np.log1p(X["capital_loss"])
        return X

    def cap_hours(X):
        X = X.copy()
        if "hours_per_week" in X.columns:
            cap = X["hours_per_week"].quantile(0.99)
            X["hours_per_week"] = X["hours_per_week"].clip(upper=cap)
        return X

    # Load Income model from Hugging Face Hub
    @st.cache_resource
    def load_income_model():
        repo_id = "nnejere/fine_tuned_income_classifier"
        pipeline_path = hf_hub_download(repo_id=repo_id, filename="full_pipeline.pkl")
        model_path = hf_hub_download(repo_id=repo_id, filename="xgb_tuned.pkl")
        preprocessor = joblib.load(pipeline_path)
        model = joblib.load(model_path)
        return preprocessor, model

    preprocessor, model = load_income_model()

    # User Input Form
    st.markdown("### 👤 Demographic Profile")
    with st.form("income_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=35)
            sex = st.selectbox("Sex", ["Male", "Female"])
            race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
            native_country = st.selectbox("Region", ["United-States", "South-America", "Asia", "Europe", "Central-America & Carribean", "North-America"])
        with col2:
            workclass = st.selectbox("Workclass", ["Unemployed", "Private", "Government", "Self-employed"])
            education = st.selectbox("Education", ["Low", "High-school", "Some-college", "Associate", "Bachelors", "Masters", "Highest"])
            occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-specialty", "Other-service", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv"])
            hours_per_week = st.slider("Hours/Week", 1, 80, 40)
        with col3:
            marital_status = st.selectbox("Marital Status", ["Married-spouse-present", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent"])
            relationship = st.selectbox("Relationship", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"])
            capital_gain = st.number_input("Capital Gain ($)", min_value=0, value=0)
            capital_loss = st.number_input("Capital Loss ($)", min_value=0, value=0)
        submit_button = st.form_submit_button("🔍 Predict Income")

    # Make Prediction
    if submit_button:
        try:
            input_data = pd.DataFrame({
                "age": [age], "workclass": [workclass], "education": [education],
                "marital_status": [marital_status], "occupation": [occupation],
                "relationship": [relationship], "race": [race], "sex": [sex],
                "capital_gain": [capital_gain], "capital_loss": [capital_loss],
                "hours_per_week": [hours_per_week], "native_country": [native_country]
            })

            # Map Sex to numeric
            input_data["sex"] = input_data["sex"].map({"Male": 0, "Female": 1})

            # Transform & predict
            processed_data = preprocessor.transform(input_data)
            prediction = model.predict(processed_data)[0]
            result = ">50K" if prediction == 1 else "<=50K"

            st.markdown("---")
            if prediction == 1:
                st.success(f"### 💡 Prediction: {result}")
                st.write("This profile indicates a high likelihood of earning over $50k.")
            else:
                st.info(f"### 💡 Prediction: {result}")
                st.write("This profile indicates a likelihood of earning $50k or less.")
        except Exception as e:
            st.error(f"An error occurred during income prediction: {e}")


# =========================
# PAGE 4: Heart Disease Prediction
# =========================
elif page == "🫀 Heart Disease Prediction":
    st.title("🫀 Heart Disease Risk Assessment")
    st.markdown("This tool uses clinical markers to predict heart disease risk.")

    # Custom preprocessing function
    def cap_outliers(df):
        df = df.copy()
        num_cols = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'st_depression']
        for col in num_cols:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    # Load Heart Disease model from Hugging Face Hub
    @st.cache_resource
    def load_heart_model():
        repo_id = "nnejere/heart_disease_classifier"
        pipeline_path = hf_hub_download(repo_id=repo_id, filename="heart_full_pipeline.pkl")
        model_path = hf_hub_download(repo_id=repo_id, filename="heart_rf_baseline.pkl")
        preprocessor = joblib.load(pipeline_path)
        model = joblib.load(model_path)
        return preprocessor, model

    preprocessor, heart_model = load_heart_model()

    # Layout for Input Form
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
            caption="Clinical Analysis",
            use_container_width=True
        )
    with col2:
        st.info("Please fill out the patient's clinical details below.")

    with st.form("heart_form"):
        st.subheader("1. Patient Vitals & History")
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=45)
            sex = st.selectbox("Sex", ["Male", "Female"])

        with c2:
            resting_bp = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=120)
            cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=220)

        with c3:
            fasting_blood_sugar = st.selectbox("Fasting BS > 120 mg/dl", ["False (<=120 mg/dl)", "True (>120 mg/dl)"])
            max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)

        st.subheader("2. Cardiac Tests & Symptoms")
        c4, c5 = st.columns(2)

        with c4:
            chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            exercise_induced_angina = st.selectbox("Ex. Induced Angina", ["No", "Yes"])

        with c5:
            st_depression = st.number_input("ST Depression", 0.0, 10.0, 1.0)
            st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
            num_major_vessels = st.selectbox("Major Vessels (0-3)", ["0","1","2","3"])
            thalassemia = st.selectbox("Thalassemia", ["Fixed Defect","Normal","Reversible Defect"])

        predict_heart = st.form_submit_button("🔍 Analyze Risk")

    # Prediction logic
    if predict_heart:
        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "chest_pain_type": chest_pain_type, "resting_bp": resting_bp,
            "cholesterol": cholesterol, "fasting_blood_sugar": fasting_blood_sugar,
            "resting_ecg": resting_ecg, "max_heart_rate": max_heart_rate,
            "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression, "st_slope": st_slope,
            "num_major_vessels": num_major_vessels, "thalassemia": thalassemia
        }])

        try:
            processed = preprocessor.transform(input_data)
            prediction = heart_model.predict(processed)[0]
            prob = heart_model.predict_proba(processed)[0][1]

            st.markdown("---")
            c1, c2 = st.columns([1,3])
            with c1:
                st.metric("Risk Probability", f"{prob*100:.1f}%")
            with c2:
                if prediction == 1:
                    st.error("### 🫀 High Risk Detected")
                    st.write("The model indicates a *high likelihood* of heart disease.")
                else:
                    st.success("### 💚 Low Risk Detected")
                    st.write("The model indicates a *low likelihood* of heart disease.")
        except Exception as e:
            st.error(f"An error occurred: {e}")










