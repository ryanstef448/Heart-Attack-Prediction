# Heart Attack Prediction: Ensemble Learning Approach

## Overview
Cardiovascular diseases are the leading cause of death globally. This project develops a predictive model to identify the risk of heart disease in patients based on clinical features like age, blood pressure, cholesterol levels, and maximum heart rate. 

The core of this project is a robust **Ensemble Voting Classifier** that aggregates predictions from multiple state-of-the-art algorithms to ensure high accuracy and reliability.

## Methodology

### 1. Data Preprocessing & Cleaning
- **Handling Physiological Zeros**: Identified and treated zero values in fields like `Cholesterol` and `RestingBP` as missing data.
- **Imputation**: Employed median imputation to handle missing values without introducing significant bias.
- **Scaling**: Used `RobustScaler` to normalize numerical features, ensuring the model is resistant to outliers.

### 2. Feature Engineering
Created advanced features to capture clinical insights:
- **Normalized Ratios**: Calculated ratios such as `CholesterolPerAge` and `RestingBPPerAge` to account for age-related physiological changes.
- **Heart Rate Metrics**: Analyzed the relationship between `MaxHR` and `RestingBP`.
- **Categorical Encoding**: Utilized One-Hot Encoding for categorical variables like `Sex` and `ChestPainType`.

### 3. Machine Learning Model (Ensemble Approach)
The project utilizes a `VotingClassifier` with soft voting, combining the strengths of:
- **Random Forest Classifier** (Calibrated)
- **Support Vector Machine (SVM)** (Calibrated)
- **Gradient Boosting Classifier** (Calibrated)
- **XGBoost**
- **LightGBM**

Probabilistic calibration was applied to the base models to ensure the predicted probabilities are representative of true likelihood.

### 4. Evaluation
- **Primary Metric**: F1-Score (to balance precision and recall).
- **Validation**: 5-fold cross-validation to ensure the model generalizes well to unseen data.

## Technologies Used
- **Python**
- **Scikit-learn** for preprocessing, pipelines, and ensemble modeling
- **XGBoost** and **LightGBM** for high-performance gradient boosting
- **Pandas** and **NumPy** for data manipulation

## Project Structure
- `Heart_Attack_Prediction.ipynb`: Well-documented Jupyter notebook containing the analysis and modeling pipeline.
- `train.csv`: Training dataset.
- `test_X.csv`: Test dataset.
- `README.md`: Project documentation.

## How to Run
1. Install requirements:
   ```bash
   pip install pandas scikit-learn xgboost lightgbm
   ```
2. Open `Heart_Attack_Prediction.ipynb` to explore the full analysis.
