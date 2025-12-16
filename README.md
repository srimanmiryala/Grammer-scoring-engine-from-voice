# 🎙️ SHL Assessment - Automated Grammar Scoring Engine

## 📄 Project Overview
This project implements a Machine Learning solution for the **SHL Intern Hiring Assessment 2025**. The goal was to build an automated scoring engine that analyzes audio recordings of spoken English and predicts a grammar proficiency score (0.0 - 5.0).

The solution utilizes advanced audio signal processing for feature extraction and a **Voting Ensemble of XGBoost Regressors** to achieve high-precision grading.

## 📊 Performance
* **Validation Metric:** RMSE (Root Mean Squared Error)
* **Validation Score:** ~0.72 (Distinction Level)
* **Leaderboard Status:** Successfully submitted.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Audio Processing:** `Librosa` (Signal analysis & feature extraction)
* **Modeling:** `XGBoost` (Gradient Boosting Trees)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Validation:** `Scikit-Learn` (K-Fold Cross Validation)

## 🧠 Methodology

### 1. Data Preprocessing & Integrity
* **Robust Indexing:** Implemented a recursive file crawler to map 400+ audio files to their absolute paths, preventing data loss from nested directory structures.
* **Audio Normalization:** All audio is re-sampled to 32kHz, trimmed of silence, and padded/cut to a fixed 5-second duration to ensure input consistency.

### 2. Feature Engineering (106 Dimensions)
We extract a dense feature vector for every audio sample to capture the nuances of speech:
* **Texture (Timbre):** MFCCs (20 coefficients), Delta, and Delta-Delta.
* **Pitch & Harmony:** Chroma features and Tonnetz (tonal centroid features).
* **Spectral Physics:** Spectral Centroid, Rolloff, Contrast, Flatness, and Zero-Crossing Rate.

### 3. Model Architecture: The "Tri-Fold" Ensemble
To prevent overfitting on the small dataset (N=409), the solution uses an ensemble of three distinct XGBoost regressors, each looking at the data differently:
* **Model A (Deep Learner):** `max_depth=7`, Low learning rate. Captures complex, non-linear speech patterns.
* **Model B (Stabilizer):** `max_depth=3`, High regularization. Prevents overfitting by focusing on broad trends.
* **Model C (Explorer):** `max_depth=5`, Random subsampling (`colsample_bytree=0.4`). forces the model to find hidden correlations in less dominant features.

**Final Prediction:** The predictions from all three models are averaged to produce the final grammar score.

## 💻 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy librosa xgboost scikit-learn tqdm
    ```

2.  **Dataset Structure:**
    Ensure the SHL dataset is placed in the root directory:
    ```text
    /dataset
       /audios (contains .wav files)
       /csvs (contains train.csv, test.csv)
    ```

3.  **Execute:**
    Run the Jupyter Notebook or Python script. The model will:
    * Index all files.
    * Extract features (with progress bar).
    * Train the ensemble using 5-Fold Cross Validation.
    * Generate `submission.csv`.

## 📜 License
This project is open-source under the MIT License.
