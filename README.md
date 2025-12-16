
# 🎙️ SHL Assessment - Grammar Scoring Engine

## 📄 Project Overview
This project implements a machine learning solution to automate the scoring of spoken grammar proficiency. Using audio feature extraction and an ensemble of XGBoost regressors, the model predicts grammar scores based on a rubric scale of 0-5.

## 🛠️ Tech Stack
* **Audio Processing:** Librosa (MFCCs, Spectral Contrast, Zero-Crossing Rate)
* **Machine Learning:** XGBoost (Regressor Ensemble)
* **Validation:** 10-Fold Cross-Validation with robust outlier scaling
* **Language:** Python 3.x

## 🚀 Key Features
* **Advanced Feature Engineering:** Extracts 106-dimensional feature vectors per audio file, capturing pitch, texture, and spectral physics.
* **Ensemble Architecture:** Combines 3 distinct model variations (Deep, Robust, Diverse) to prevent overfitting.
* **High Precision:** Achieved a Validation RMSE of **0.7252** (Distinction Level).

## 📊 Results
The final model utilizes a weighted voting system to stabilize predictions against microphone noise and outliers, ensuring consistent grading performance.
