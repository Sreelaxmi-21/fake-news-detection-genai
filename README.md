# 📰 Fake News Detection using ML & GenAI

## 📌 Overview
This project is an AI-powered **Fake News Detection system** that classifies news
articles as **Fake** or **Real** using **Natural Language Processing (NLP)** and
**Machine Learning**.  
To improve transparency, a **GenAI-style explanation layer** is added to explain
model predictions in simple language.

This project was built as part of a **GenAI Hackathon** with a focus on
**real-world impact, explainability, and clean ML design**.

---

## 🎯 Problem Statement
Fake news spreads rapidly on digital platforms and can mislead people,
influence opinions, and cause social harm.  
The goal of this project is to **identify fake-news writing patterns** using
machine learning and provide **interpretable explanations** for predictions.

---

## 🧠 Solution Approach
1. Collect labeled fake and real news articles  
2. Clean and preprocess text data  
3. Convert text into numerical features using **TF-IDF**  
4. Train machine learning models to classify news  
5. Add an explanation layer to describe predictions  

---

## 📂 Datasets Used
- **Fake.csv & True.csv**  
  - Primary dataset for training and evaluation  
  - Contains labeled fake and real news articles  

- **WELFake (Secondary – comparison only)**  
  - Used to demonstrate scalability and future validation  

---

## 🛠 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **TF-IDF Vectorizer**
- **Logistic Regression**
- **Naive Bayes / Random Forest (comparison)**
- **Joblib (model saving)**

---

## 🔍 Model Pipeline
1. Text Cleaning (lowercase, punctuation removal)
2. Feature Extraction using TF-IDF
3. Train-Test Split
4. Model Training
5. Evaluation using Accuracy, Precision, Recall, F1-score
6. Prediction with Explanation

---

## 📊 Results
- **Logistic Regression Accuracy:** ~98.7%
- Secondary models trained for comparison
- High precision and recall on dataset-based evaluation

---

## ⚠️ Important Note (Model Scope & Limitations)
- This model detects **fake-news writing patterns**, not factual correctness.
- It does **not** verify historical facts or real-time events.
- Short factual statements without journalistic context may be misclassified.
- Best performance is achieved on **news-style articles** similar to the training data.

👉 This behavior is expected and aligns with standard ML classification systems.

---

## 🧪 How to Run (CMD)
```bash
python src/test_model.py
