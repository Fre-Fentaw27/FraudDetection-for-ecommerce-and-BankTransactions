# FraudDetection-for-ecommerce-and-BankTransactions

Improved detection of fraud cases for e-commerce and bank transactions

# Task 1: Data Analysis & Preprocessing

s
**Adey Innovation Inc.**

## 📌 **Project Overview**

This project focuses on improving fraud detection for e-commerce and banking transactions. Task 1 covers **data analysis, preprocessing, and feature engineering** to prepare datasets for model training.

### 📂 **Dataset Sources**

1. `Fraud_Data.csv`: E-commerce transactions (user behavior, device, IP, purchase details).
2. `IpAddress_to_Country.csv`: Maps IP ranges to countries.
3. `creditcard.csv`: Bank transactions with anonymized features (V1-V28).

---

## 🛠️ **Task 1 Implementation**

### **1. Data Processing**

**Script**: [`src/data_processing.py`](src/data_processing.py)

- Handles missing values (imputation/dropping).
- Removes duplicates and corrects data types (e.g., datetime conversion).

### **2. Exploratory Data Analysis (EDA)**

**Script**: [`src/eda.py`](src/eda.py) | **Notebook**: [`notebooks/plots/`](notebooks/plots/)

- **Univariate Analysis**: Class imbalance, transaction value distributions.
- **Bivariate Analysis**: Fraud vs. time, geolocation, purchase patterns.
- **Sample Outputs**
  - ![Amount vs Fraud](notebooks/plots/fraud_data_country_analysis.png)
  - ![Amount vs Fraud](notebooks/plots/credit_data_day_of_week_analysis.png)

### **3. Feature Engineering**

**Script**: [`src/data_engineering.py`](src/data_engineering.py)

- **Time-Based Features**: `hour_of_day`, `day_of_week`, `time_since_signup`.
- **Behavioral Features**: Transaction frequency/velocity.
- **Geolocation**: Merged IP addresses with country data.

### **4. Data Transformation**

**Notebook**: [`notebooks/data_transformation.ipynb`](notebooks/data_transformation_1.ipynb)

- **Class Imbalance**: Applied SMOTE (oversampling) on training data.
- **Scaling**: Standardized numerical features (`StandardScaler`).
- **Encoding**: One-Hot for categorical variables (`source`, `browser`, `country`).

---

## 📊 **Key Findings**

✅ Fraud rate: **2.5%** (highly imbalanced).  
✅ High-value transactions (>$500) are 5x more likely to be fraudulent.  
✅ Peak fraud times: **8 PM–2 AM**, especially on weekends.

---

## 🚧 **Challenges**

1. **IP-to-Country Merge**: Required integer conversion and interval matching.
2. **Class Imbalance**: SMOTE improved recall but needed careful validation.

---

## 🚀 **Next Steps**

- Proceed to **Task 2 (Model Training & Evaluation)**.
- Focus on optimizing recall to minimize false negatives.

---

# Task 2: Model Building and Training for Fraud and Credit Card Datasets

This document details the second phase of the project: building, training, and evaluating machine learning models on the preprocessed Fraud and Credit Card Transaction datasets. The primary goal is to compare a simple, interpretable model (Logistic Regression) with a powerful ensemble model (Random Forest) for both classification tasks, focusing on metrics suitable for imbalanced datasets.

## Table of Contents

1.  [Project Structure](#project-structure)
2.  [Data Preparation (Recap from Task 1)](#data-preparation-recap-from-task-1)
3.  [Model Selection](#model-selection)
4.  [Model Training and Evaluation](#model-training-and-evaluation)
    - [Fraud Data Performance Summary](#fraud-data-performance-summary)
    - [Credit Data Performance Summary](#credit-data-performance-summary)
5.  [Justification of Best Model](#justification-of-best-model)
    - [For Fraud Data](#for-fraud-data)
    - [For Credit Data](#for-credit-data)
6.  [Conclusion and Future Work](#conclusion-and-future-work)
7.  [How to Run the Code](#how-to-run-the-code)

---

## 1. Project Structure

The relevant files for this task are:

- `notebooks/task2_model_building.ipynb`: The Jupyter Notebook containing the code for model training and evaluation.
- `data/processed_for_modeling/`: Directory where the preprocessed and saved datasets from Task 1 are stored.
  - `X_train_fraud.npz`, `X_test_fraud.npz`: Sparse feature matrices for fraud data.
  - `y_train_fraud.csv`, `y_test_fraud.csv`: Target labels for fraud data.
  - `X_train_credit.npy`, `X_test_credit.npy`: Dense feature arrays for credit data.
  - `y_train_credit.csv`, `y_test_credit.csv`: Target labels for credit data.

## 2. Data Preparation (Recap from Task 1)

Before model building, the raw datasets underwent significant preprocessing as detailed in Task 1. Key steps included:

- **Feature-Target Separation and Train-Test Split:** The datasets were split into features (`X`) and target (`y`), and then further into training and testing sets to ensure robust model evaluation. Stratified splitting was used to preserve class proportions.
- **Handling Imbalance:**
  - **Fraud Data:** Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training set to balance the highly imbalanced fraud classes.
  - **Credit Data:** Random Undersampling was applied to the training set to balance the credit default classes.
- **Feature Scaling:** Numerical features were standardized using `StandardScaler`.
- **Categorical Encoding:** High-cardinality categorical features were encoded using `TargetEncoder` to reduce dimensionality and avoid memory issues that would arise from One-Hot Encoding large numbers of categories.

Crucially, due to the high dimensionality of the fraud dataset, its feature matrices (`X_train_fraud`, `X_test_fraud`) were loaded and kept as **SciPy sparse CSR matrices (`.npz` files)** to manage memory efficiently. The credit dataset's feature matrices (`X_train_credit`, `X_test_credit`) were loaded as **dense NumPy arrays (`.npy` files)**.

## 3. Model Selection

For each dataset, two types of classification models were chosen for comparison:

1.  **Logistic Regression:**

    - **Rationale:** Chosen as a simple, interpretable baseline model. Its linear nature helps understand basic relationships, and it serves as a good benchmark against more complex models. The `liblinear` solver was selected for its efficiency with sparse data (where applicable).

2.  **Random Forest Classifier:**
    - **Rationale:** Chosen as a powerful ensemble model. Random Forests are robust, handle non-linear relationships, are less prone to overfitting than individual decision trees, and can effectively work with high-dimensional data, including sparse input. `n_jobs=-1` was used to leverage all available CPU cores for faster training.

## 4. Model Training and Evaluation

Both models were trained on the respective preprocessed training sets and evaluated on their corresponding test sets. Given the inherent class imbalance in both fraud and credit default prediction, specific metrics were prioritized:

- **Average Precision (AUC-PR):** Area Under the Precision-Recall Curve. This metric is highly recommended for imbalanced datasets as it focuses on the performance of the positive class and the trade-off between precision and recall, providing a more informative assessment than AUC-ROC.
- **F1-Score:** The harmonic mean of precision and recall. It balances these two metrics and is a good overall indicator when both false positives and false negatives are important.
- **Confusion Matrix:** Provides a detailed breakdown of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN), crucial for understanding the types of errors made.
- **Classification Report:** Offers precision, recall, and F1-score for each class, along with support.

### Fraud Data Performance Summary

| Metric                         | Logistic Regression | Random Forest |
| :----------------------------- | :------------------ | :------------ |
| **Average Precision (AUC-PR)** | **0.6465**          | 0.6264        |
| **F1-Score**                   | 0.6963              | **0.6991**    |
| **Precision (Class 1)**        | 0.9879              | **0.9996**    |
| **Recall (Class 1)**           | **0.5376**          | **0.5376**    |
| **Confusion Matrix (TP)**      | 2282                | 2282          |
| **Confusion Matrix (FP)**      | 28                  | **1**         |
| **Confusion Matrix (FN)**      | 1963                | 1963          |
| **Confusion Matrix (TN)**      | 41061               | 41088         |

_(Class 1 refers to the fraudulent class)_

### Credit Data Performance Summary

| Metric                         | Logistic Regression | Random Forest |
| :----------------------------- | :------------------ | :------------ |
| **Average Precision (AUC-PR)** | 0.4857              | **0.6981**    |
| **F1-Score**                   | 0.0638              | **0.0940**    |
| **Precision (Class 1)**        | 0.0331              | **0.0497**    |
| **Recall (Class 1)**           | **0.8873**          | 0.8732        |
| **Confusion Matrix (TP)**      | 126                 | 124           |
| **Confusion Matrix (FP)**      | 3679                | **2372**      |
| **Confusion Matrix (FN)**      | **16**              | 18            |
| **Confusion Matrix (TN)**      | 81297               | 82604         |

_(Class 1 refers to the default class)_

## 5. Justification of Best Model

### For Fraud Data

The 'best' model choice for fraud detection requires a careful trade-off between identifying true fraud (recall) and minimizing false alarms (precision/false positives).

- **Logistic Regression** achieved a slightly higher **Average Precision (AUC-PR)** of 0.6465 compared to Random Forest's 0.6264. This indicates that Logistic Regression generally provides a better ranking of positive instances across different recall levels.
- However, examining the **Confusion Matrix and Classification Report** reveals a critical insight:
  - Both models achieved an **identical Recall of 0.5376** for the positive (fraud) class, meaning they correctly identified the same number of fraudulent transactions (2282 TPs) and missed the same number (1963 FNs).
  - **Random Forest** achieved a near-perfect **Precision of 0.9996** for the fraud class, significantly higher than Logistic Regression's 0.9879. This is due to Random Forest identifying only **1 False Positive**, whereas Logistic Regression had 28 False Positives.
  - While Random Forest's AUC-PR is slightly lower, its ability to dramatically reduce false positives (incorrectly flagging legitimate transactions as fraud) is a massive advantage in real-world fraud detection. A false positive often leads to manual review, customer inconvenience, and operational costs.
  - Despite the slightly lower AUC-PR, the **Random Forest's F1-Score (0.6991)** is marginally higher than Logistic Regression's (0.6963), further reinforcing its slightly better overall balance given the very high precision.

**Conclusion for Fraud Data:** Given that minimizing false positives is often a key business objective in fraud detection (to avoid customer friction and wasted resources), and the recall is identical, **Random Forest is arguably the better model for the Fraud Data.** While Logistic Regression has a slightly higher AUC-PR, Random Forest's significantly lower False Positive rate (1 vs 28) makes it highly valuable. The nearly perfect precision is a strong indicator of its reliability when it flags a transaction as fraudulent.

### For Credit Data

The 'best' model is clearly **Random Forest**.

- It achieved a substantially higher **Average Precision (AUC-PR) of 0.6981** compared to Logistic Regression's 0.4857. This demonstrates Random Forest's superior ability to differentiate between defaulting and non-defaulting customers across various thresholds.
- Furthermore, Random Forest yielded a significantly better **F1-Score (0.0940)** against Logistic Regression's 0.0638, indicating a more effective balance between precision and recall in predicting defaults.
- While both models achieve high recall (around 87-88%), indicating they are good at identifying most true defaults, they both suffer from very low precision (many False Positives). However, Random Forest's precision (0.0497) is still notably better than Logistic Regression's (0.0331), leading to fewer individuals being incorrectly flagged as high risk.

**Conclusion for Credit Data:** Random Forest is the superior choice for this dataset based on all key metrics, including AUC-PR, F1-Score, and a better balance of precision/recall. It is better at leveraging the complexity of the features to make more accurate predictions.

## 6. Conclusion and Future Work

This task successfully demonstrated the training and evaluation of two distinct models on both datasets, with careful consideration for class imbalance.

**Key Takeaways:**

- For **Fraud Data**, **Random Forest** showed a critical advantage in minimizing false positives, making it a more practical choice despite a slightly lower AUC-PR than Logistic Regression.
- For **Credit Data**, **Random Forest** clearly outperformed Logistic Regression across all key imbalance-aware metrics.

**Future Work:**

- **Hyperparameter Tuning:** The current models use default parameters. Significant performance gains can be expected by rigorously tuning hyperparameters (e.g., using `GridSearchCV` or `RandomizedSearchCV`).
- **Advanced Ensemble Models:** Exploring other powerful ensemble methods like LightGBM or XGBoost, which are highly optimized for performance and large sparse datasets, could yield even better results.
- **Feature Engineering:** Further domain-specific feature engineering could enhance the predictive power of the models.
- **Threshold Optimization:** For both datasets, especially credit, the very low precision indicates that the default classification threshold (0.5) might not be optimal. Adjusting the probability threshold based on business costs (e.g., higher cost of false negatives vs. false positives) could improve real-world utility.
- **More Sophisticated Imbalance Techniques:** Investigating more advanced imbalanced learning techniques or custom loss functions tailored to specific misclassification costs.

## 7. How to Run the Code

1.  **Prerequisites:** Ensure you have Python 3.x and the necessary libraries installed (`pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `category_encoders`, `imblearn`). You can install them via pip:
    ```bash
    pip install pandas numpy scikit-learn scipy matplotlib seaborn category_encoders imblearn
    # If using LightGBM or XGBoost:
    # pip install lightgbm xgboost
    ```
2.  **Data:** Ensure you have completed Task 1 and the preprocessed data is saved in the `data/processed_for_modeling/` directory as specified.
3.  **Navigate:** Open the `notebooks/task2_model_building.ipynb` Jupyter Notebook.
4.  **Execute:** Run all cells in the notebook sequentially. The output will include performance summaries, confusion matrices, and precision-recall curves, culminating in the justification of the best models.
