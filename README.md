# Credit Card Default Prediction & Risk Classification

Analisis prediksi default kartu kredit menggunakan machine learning untuk mengidentifikasi kemungkinan nasabah mengalami gagal bayar.

## Dataset

Dataset yang digunakan adalah **UCI Credit Card Default Dataset** yang berisi informasi nasabah kartu kredit dan status default pembayaran.

## Data Preprocessing

Tahapan preprocessing yang dilakukan meliputi:

- Data exploration
- Missing value checking
- Duplicate checking
- Outlier detection
- Outlier capping
- Data normalization
- Categorical encoding
- Train-test splitting

## Feature Engineering

Dilakukan pembentukan beberapa fitur baru berdasarkan informasi tagihan dan pembayaran, seperti:

- Average bill amount
- Average payment amount
- Payment ratio
- Maximum utilization
- Mean payment delay
- Log-transformed credit limit

## Machine Learning

Tiga algoritma klasifikasi digunakan:

1. Logistic Regression
2. Decision Tree
3. XGBoost

Model kemudian dibandingkan dan dilakukan hyperparameter tuning untuk memperoleh performa yang lebih baik.

## Model Evaluation

Evaluasi model menggunakan:

- Recall
- Balanced Accuracy
- F1 Macro
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Risk Classification

Predicted probability dari model digunakan untuk melakukan segmentasi risiko nasabah menjadi:

- Low Risk
- Medium Risk
- High Risk

## Tools

- R
- dplyr
- ggplot2
- caret
- glmnet
- xgboost
- pROC
- tidyr

## Results

Hasil analisis mencakup perbandingan performa tiga model klasifikasi, evaluasi menggunakan berbagai metrik, serta segmentasi nasabah berdasarkan probabilitas default.
