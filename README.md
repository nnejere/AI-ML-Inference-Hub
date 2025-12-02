# 🤖 AI & ML Inference Hub

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-green)](https://streamlit.io/)  
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/index)  
[![Machine Learning](https://img.shields.io/badge/ML-Python-red)](https://www.python.org/)  

---

## 📝 Project Overview

The **AI & ML Inference Hub** is a **multi-purpose interactive web platform** built with **Streamlit** for real-time inference using advanced AI and Machine Learning models. The platform integrates multiple ML pipelines, enabling users to explore and interact with:

1. **🎬 Sentiment Analysis:** Classifies IMDb movie reviews as positive or negative using a fine-tuned **DistilBERT** model.
2. **💼 Income Prediction:** Predicts whether an individual's income exceeds $50K/year using a tabular ML pipeline with preprocessing and a trained model.
3. **🫀 Heart Disease Prediction:** Assesses heart disease risk using clinical and demographic features with a trained ML classifier.

The platform is designed for **interactive exploration, testing, and demonstration** of AI/ML models in real-time.

---

## 🎯 Objectives

- Provide a unified interface for multiple AI/ML models in one web application.
- Enable **interactive, real-time predictions** for both text and tabular datasets.
- Demonstrate **end-to-end ML pipelines**, including preprocessing, model inference, and result visualization.
- Facilitate educational and practical exploration of **state-of-the-art models** for NLP and supervised tabular tasks.

---

## 🛠️ Technical Workflow

### 1. Streamlit Application Structure
- **Sidebar Navigation:** Switch between projects and overview page.
- **Dynamic Content:** Pages dynamically load forms, inputs, and model predictions.
- **Custom Styling:** Optimized buttons, container padding, and UI elements.

### 2. Sentiment Analysis Module
- **Model:** Fine-tuned **DistilBERT**
- **Input:** Text review (max 128 tokens)
- **Processing:** Tokenization and inference via Hugging Face `AutoModelForSequenceClassification`
- **Output:** Predicted sentiment (Positive/Negative) with visual cues

### 3. Income Prediction Module
- **Data:** Tabular demographic and financial features
- **Preprocessing:** Custom transformers for scaling, one-hot encoding, log transformations
- **Model:** Trained XGBoost classifier
- **Output:** Predicted income bracket (`>50K` or `<=50K`) with probability

### 4. Heart Disease Prediction Module
- **Data:** Clinical features (age, BP, cholesterol, ECG, etc.)
- **Preprocessing:** Outlier clipping and categorical encoding
- **Model:** Pre-trained Random Forest classifier
- **Output:** Predicted risk (Low/High) with probability

### 5. Model Loading
- Models are downloaded from **Hugging Face Hub** using `hf_hub_download` for reproducibility.
- Includes Sentiment Analysis, Income Prediction, and Heart Disease Prediction models.

### 6. Session State Management
- Tracks the current page across user interactions
- Provides smooth navigation between modules

---

## 📊 Key Features

- Multi-model interactive interface
- Real-time inference for text and tabular data
- Preprocessing pipelines integrated in-app
- Visual feedback (metrics, emojis, success/error messages, balloons)
- Deployment-ready for **educational and demonstration purposes**

---

## 🔹 Recommendations

- **Expand Supported Models:** Add more NLP or tabular ML models for broader coverage.
- **Continuous Deployment:** Integrate with Hugging Face Hub or cloud services for automatic updates.
- **User Feedback Loop:** Collect input to improve models and datasets.
- **Add Monitoring:** Track predictions and app usage for evaluation.
- **Interpretability Tools:** Consider LIME or SHAP for explaining model predictions.
- **Hyperparameter Optimization:** Further tuning can improve model robustness and accuracy.

---

## 🏁 Conclusion

The **AI & ML Inference Hub** demonstrates the power of integrating multiple ML pipelines into a single, user-friendly platform. Key takeaways:

- End-to-end pipelines for **text and tabular ML tasks**
- Seamless **real-time inference and interactive exploration**
- Clear demonstration of **preprocessing, model inference, and visualization**
- Ready for deployment and educational demonstration
- Provides a foundation for **extending AI/ML applications** for practical use cases like sentiment analysis, income prediction, and health risk assessment

---

## 🚀 How to Run This Project Locally

### Prerequisites
- Python 3.9+
- Streamlit
- GPU recommended for NLP inference (optional)
- Access to Hugging Face Hub for model downloads

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install Dependencies
pip install -r requirements.txt
# Or manually:
pip install streamlit transformers torch scikit-learn pandas numpy huggingface_hub joblib

3. Run the application
   streamlit run app.py

