# Stroke Prediction

## Overview
This project predicts the probability of a stroke based on health attributes using machine learning techniques. The dataset is sourced from Kaggle and includes features such as age, BMI, glucose level, smoking status, and medical history.

## Dataset
The dataset contains the following attributes:
- **gender**: Male, Female, or Other
- **age**: Age of the patient
- **hypertension**: 0 (No) or 1 (Yes)
- **heart_disease**: 0 (No) or 1 (Yes)
- **ever_married**: Yes or No
- **work_type**: Type of employment
- **Residence_type**: Rural or Urban
- **avg_glucose_level**: Average glucose level in the blood
- **bmi**: Body Mass Index
- **smoking_status**: Formerly smoked, never smoked, smokes, or unknown
- **stroke**: 0 (No stroke) or 1 (Stroke) - Target variable

## Data Preprocessing
- **Handling Missing Values**: Replaced missing values in categorical and numerical columns using mode and median imputation.
- **Encoding**: One-hot encoding applied to categorical features.
- **Outlier Detection**: Adjusted BMI values greater than 50.
- **Feature Selection**: Removed weakly correlated features.
- **Data Balancing**: Addressed class imbalance in the dataset.

## Model Training
The `BalancedRandomForestClassifier` from the `imblearn` package was used to handle the imbalanced dataset. The pipeline includes:
1. Splitting the dataset into training and test sets.
2. Training the model on the balanced dataset.
3. Evaluating the model using accuracy and a classification report.

## Results
The model achieved a test accuracy of approximately **[insert accuracy value]**. Further improvements can be explored using feature engineering and hyperparameter tuning.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn

## How to Run
1. Download the dataset using:
!kaggle datasets download fedesoriano/stroke-prediction-dataset !unzip /content/stroke-prediction-dataset.zip

2. Run the `stroke_prediction_.py` script.
   
3. The model will be trained, and evaluation metrics will be displayed.

## Future Improvements
- Experiment with other classification models such as XGBoost or Neural Networks.
- Perform hyperparameter tuning for better accuracy.
- Apply feature engineering to improve prediction quality.

## Acknowledgments
Dataset sourced from Kaggle. This project is for educational purposes.
نسخ
تحرير
