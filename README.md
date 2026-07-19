# 🎓 AI-Driven Student Performance Prediction System

## 📌 Project Description

The AI-Driven Student Performance Prediction System is a Machine Learning application that predicts a student's final exam score and overall academic performance based on academic, personal, and behavioral factors.

The project uses three Machine Learning techniques:

- Linear Regression
- Logistic Regression
- Principal Component Analysis (PCA)

A Streamlit web application provides an interactive interface where users can enter student details and receive predictions instantly.

## 🎯 Objectives

- Predict students' Final Exam Scores.
- Classify students into performance categories.
- Reduce feature dimensionality using PCA.
- Build an interactive web application using Streamlit.
- Analyze student data through visualizations.

- ## ✨ Features

- Predict Final Exam Score
- Predict Student Performance Category
- Interactive Prediction Form
- Data Visualization Dashboard
- Model Performance Metrics
- User-Friendly Interface
- Machine Learning Model Integration

- ## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- Joblib

- ## 🤖 Machine Learning Algorithms

### Linear Regression

Used to predict the student's Final Exam Score.

### Logistic Regression

Used to classify students into:

- Poor
- Average
- Good
- Excellent

### Principal Component Analysis (PCA)

Used for dimensionality reduction to improve data processing efficiency.

## 📂 Dataset

Dataset Name:

Student Academic Performance Dataset

Dataset Size:

- 500,000 Records
- 27 Original Features

After preprocessing:

- 46 Features

- ## 📁 Project Structure

```text
student-performance-prediction/
│
├── app/
│   └── main.py
│
├── dataset/
│   └── new.csv
│
├── models/
│   ├── linear_model.pkl
│   ├── logistic_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── student_prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/architvishnoi03/AI-Driven-Student-Performance-Prediction-System.git
```

Go to the project folder

```bash
cd AI-Driven-Student-Performance-Prediction-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
cd app
python -m streamlit run main.py
```

## 📱 Application Pages

### 🏠 Home

Project overview and objectives.

### 🔮 Predict

Predicts:

- Final Exam Score
- Student Performance Category

### 📊 Visualization

Displays:

- Performance Distribution
- Exam Score Distribution
- Correlation Heatmap

### 📈 Model Performance

Displays:

- Linear Regression R² Score
- Linear Regression MSE
- Logistic Regression Accuracy

### ℹ️ About

Project information and developer details.

## 📊 Model Performance

### Linear Regression

- Mean Squared Error (MSE): **56.56**
- R² Score: **0.5573**

### Logistic Regression

- Accuracy: **74.14%**

## 🚀 Future Improvements

- Improve Logistic Regression convergence.
- Deploy the application online.
- Add more Machine Learning models for comparison.
- Improve prediction accuracy through feature engineering.
- Add real-time student analytics.

## 👨‍💻 Developed By

**Archit Vishnoi**

B.Tech Computer Science Engineering

Dr. A.P.J. Abdul Kalam Technical University (AKTU)

GitHub:
https://github.com/architvishnoi03

## 📜 License

This project is developed for educational and learning purposes.
