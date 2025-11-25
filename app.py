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


# Set page configuration
st.set_page_config(
    page_title="ML Inference Hub", 
    layout="wide",
    page_icon="🤖"
)

# --- CUSTOM STYLING ---
# This injects custom CSS to style buttons and headers to look more professional
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


# Sidebar for navigation
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1518932945647-7a1c969f8be2?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="ML Hub v1.0")
    st.title("🔍 Navigation")
    page = st.radio("Select a project:", ["🏠 Overview", "🎬 Sentiment Analysis", "💼 Income Prediction", "❤️ Heart Disease Prediction"])
    st.markdown("---")
    st.caption("Powered by Streamlit & Transformers")


# --- Page 1: Overview ---
if page == "🏠 Overview":
    st.title("Welcome to the AI & ML Inference Hub 🤖")
    
    # Hero Image - Updated to a "Digital Brain/AI" concept
    st.image("https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    
    st.markdown("### Choose a Project to Explore From The Navigation SideBar")
    st.write("This platform brings together advanced machine learning models for interactive testing.")

    # Create 3 columns for a "Card" look
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎬 Sentiment Analysis")
        st.write("Analyze IMDB movie reviews to detect positive or negative sentiment using DistilBERT.")
    
    with col2:
        st.subheader("💼 Income Prediction")
        st.write("Predict if an individual earns >$50K based on census demographic data.")

    with col3:
        st.subheader("❤️ Heart Disease")
        st.write("Assess medical risk factors to predict the likelihood of heart disease.")


# --- Page 2: Sentiment Analysis ---
elif page == "🎬 Sentiment Analysis":
    st.title("🎬 Sentiment Analysis")
    st.markdown("### IMDB Movie Review Classifier")
    
    col_main, col_img = st.columns([2, 1])
    
    with col_img:
        st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", use_container_width=True)

    with col_main:
        # Load model and tokenizer once
        @st.cache_resource
        def load_model():
            model_path = os.path.join(os.path.dirname(__file__), "fine_tuned_distilbert_imdb")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return tokenizer, model
        
        tokenizer, model = load_model()

        text = st.text_area("📝 Enter a movie review below:", height=150, placeholder="This movie was fantastic! The plot was intriguing...")

        if st.button("Analyze Sentiment"):
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


# --- Page 3: Income Prediction ---
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

    # Load preprocessing pipeline and model
    @st.cache_resource
    def load_income_model():
        base_dir = os.path.join(os.path.dirname(__file__), "fine_tuned_income_classifier")
        pipeline_path = os.path.join(base_dir, "full_pipeline.pkl")
        model_path = os.path.join(base_dir, "xgb_tuned.pkl")
        preprocessor = joblib.load(pipeline_path)
        model = joblib.load(model_path)
        return preprocessor, model

    preprocessor, model = load_income_model()

    # ---  Input Layout ---
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

    if submit_button:
        # Create DataFrame
        input_data = pd.DataFrame({
            "age": [age], "workclass": [workclass], "education": [education],
            "marital_status": [marital_status], "occupation": [occupation],
            "relationship": [relationship], "race": [race], "sex": [sex],
            "capital_gain": [capital_gain], "capital_loss": [capital_loss],
            "hours_per_week": [hours_per_week], "native_country": [native_country]
        })

        input_data["sex"] = input_data["sex"].map({"Male": 0, "Female": 1})
        
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


# --- Page 4: Heart Disease Prediction ---
elif page == "❤️ Heart Disease Prediction":
    st.title("❤️ Heart Disease Risk Assessment")
    st.markdown("This tool uses clinical markers to predict heart disease risk.")

    def cap_outliers(df):
        df = df.copy()
        num_cols = ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'st_depression']
        for col in num_cols:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    @st.cache_resource
    def load_heart_model():
        base_path = os.path.join(os.path.dirname(__file__), "heart_disease_classifier")
        pipeline_path = os.path.join(base_path, "heart_full_pipeline.pkl")
        model_path = os.path.join(base_path, "heart_rf_baseline.pkl")
        preprocessor = joblib.load(pipeline_path)
        model = joblib.load(model_path)
        return preprocessor, model

    preprocessor, heart_model = load_heart_model()

    # --- Organized Layout ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80", caption="Clinical Analysis", use_container_width=True)
    
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
            num_major_vessels = st.selectbox("Major Vessels (0-3)", ["0", "1", "2", "3"])
            thalassemia = st.selectbox("Thalassemia", ["Fixed Defect", "Normal", "Reversible Defect"])

        predict_heart = st.form_submit_button("🔍 Analyze Risk")

    if predict_heart:
        input_data = pd.DataFrame([{
            "age": age, "sex": sex, "chest_pain_type": chest_pain_type,
            "resting_bp": resting_bp, "cholesterol": cholesterol,
            "fasting_blood_sugar": fasting_blood_sugar, "resting_ecg": resting_ecg,
            "max_heart_rate": max_heart_rate, "exercise_induced_angina": exercise_induced_angina,
            "st_depression": st_depression, "st_slope": st_slope,
            "num_major_vessels": num_major_vessels, "thalassemia": thalassemia
        }])

        try:
            processed = preprocessor.transform(input_data)
            prediction = heart_model.predict(processed)[0]
            probability = heart_model.predict_proba(processed)[0][1]

            st.markdown("---")
            
            # Using Metrics for nicer display
            c_res1, c_res2 = st.columns([1,3])
            with c_res1:
                st.metric(label="Risk Probability", value=f"{probability*100:.1f}%")
            
            with c_res2:
                if prediction == 1:
                    st.error(f"### ❤️ High Risk Detected")
                    st.write("The model suggests a high likelihood of heart disease.")
                else:
                    st.success(f"### 💚 Low Risk Detected")
                    st.write("The model suggests a low likelihood of heart disease.")

        except Exception as e:
            st.error(f"An error occurred: {e}")