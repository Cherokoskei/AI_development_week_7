# Predicting Student Dropout Rates

##  Problem Statement
This project aims to develop a machine learning model to predict student dropout risk based on demographic, academic, and financial data.

##  Objectives
- Build a classification model to identify students at risk of dropping out.
- Understand key factors contributing to student dropout.
- Enable institutions to intervene early and improve retention rates.

##  Stakeholders
- University Administrators
- Academic Advisors

## Evaluation Metrics
- **Accuracy**: Measures the overall correctness of predictions.
- **F1-Score**: Balances precision and recall, useful when classes are imbalanced.

##  Dataset
- **Source**: Kaggle – *Predict Students' Dropout and Academic Success*
- **Path**: `data/raw/student_dropout.csv`

## ⚙️ Modeling & Deployment
- **Model**: Random Forest Classifier
- **Preprocessing**: Handling missing values, encoding categorical variables, scaling
- **Deployment**: Trained model stored in `models/`, can be loaded and used for prediction
