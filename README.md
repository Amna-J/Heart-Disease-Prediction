# Heart-Disease-Prediction

This project focuses on building a machine learning model to predict the presence of heart disease in patients based on various medical attributes. Using the Heart Disease UCI dataset, we perform data cleaning, exploratory data analysis (EDA), and train a Logistic Regression classifier to assess heart disease risk.

Objective

To analyze medical factors contributing to heart disease.

To build a binary classification model (Logistic Regression) that predicts whether a patient has heart disease (Target = 1) or not (Target = 0).

To evaluate the model's performance using standard metrics.

Dataset

Source: Heart Disease UCI Dataset (Kaggle)

Filename: HeartDiseaseTrain-Test.csv

Key Features:

age: Age of the patient

sex: Gender

cp: Chest pain type (4 values)

trestbps: Resting blood pressure

chol: Serum cholestoral in mg/dl

fbs: Fasting blood sugar > 120 mg/dl

restecg: Resting electrocardiographic results

thalach: Maximum heart rate achieved

exang: Exercise induced angina

oldpeak: ST depression induced by exercise relative to rest

slope: The slope of the peak exercise ST segment

ca: Number of major vessels (0-3) colored by flourosopy

thal: Thalassemia

target: 0 = No Disease, 1 = Disease

Methodology

1. Data Loading & Cleaning

Loaded the dataset using Pandas.

Checked for missing values (Dataset was found to be clean with 0 missing values).

Verified data types and structure.

2. Exploratory Data Analysis (EDA)

Target Distribution: Analyzed the balance between healthy and diseased patients.

Correlation Matrix: Visualized relationships between numerical features to identify multicollinearity and strong predictors.

3. Data Preprocessing

Encoding: Converted categorical variables (e.g., sex, chest_pain_type, thalassemia) into numerical values using Label Encoding.

Splitting: Divided data into training (80%) and testing (20%) sets.

Scaling: Applied Standard Scaling to normalize feature values for optimal Logistic Regression performance.

4. Model Training

Algorithm: Logistic Regression.

Training: Fitted the model on the scaled training data.

5. Evaluation

The model was evaluated on the test set with the following results:

Accuracy: 80.49%

ROC-AUC Score: 0.88 (Indicates strong separability between classes)

Confusion Matrix: Generated to visualize True Positives, False Positives, True Negatives, and False Negatives.

6. Feature Importance

Identified the top risk factors driving the model's predictions:

Vessels colored by flourosopy

Thalassemia

Chest Pain Type

Sex

Oldpeak

How to Run

Prerequisites: Ensure Python is installed along with the following libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Dataset: Place HeartDiseaseTrain-Test.csv in the project directory.

Execution: Run the main script (e.g., in a Jupyter Notebook or Python file):

python main.py


Visualizations

The project generates the following plots:

eda_target.png: Distribution of target classes.

eda_corr.png: Heatmap of feature correlations.

confusion_matrix.png: Model classification performance.

roc_curve.png: ROC curve with AUC score.

feature_importance.png: Bar chart of model coefficients.
