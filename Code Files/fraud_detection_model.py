# Required Imports
# Pandas
import pandas as pd
# Numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
# Seaborn
import seaborn as sns

# Preprocessing, pipeline, compose, ensemble , impute and other imports from Scikit Learn
# Scikit Learn
from sklearn.model_selection import train_test_split
# Standard  Scaler import, OneHotEncoder Import
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Column Transformer
from sklearn.compose import ColumnTransformer
# Pipeline
from sklearn.pipeline import Pipeline
# SimpleImputer
from sklearn.impute import SimpleImputer
# Random Forest Classifier Import
from sklearn.ensemble import RandomForestClassifier
# Scikit Learn Martix 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import from SMOTE
from imblearn.over_sampling import SMOTE

# Load the dataset
# Dataset import that contain data for training and testing purposes
data = pd.read_csv('FastagFraudDetection.csv')

# Data exploration
print(data.info())
# describe
print(data.describe())
# Value Counts
print(data['Fraud_indicator'].value_counts())

# Data
sns.countplot(x='Fraud_indicator', data=data)
# Show
plt.show()

# Data preprocessing
# Preprocessing Fastag Id, Transaction Id and other details
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%d/%Y %H:%M')
# Hours
data['Hour'] = data['Timestamp'].dt.hour
# Days of Week Data
data['Day_of_Week'] = data['Timestamp'].dt.dayofweek
# Vehicle details, transaction details, and timings
data = data.drop(columns=['Transaction_ID', 'Timestamp', 'Vehicle_Plate_Number'])
# Fastag details
data['FastagID'] = data['FastagID'].fillna('Unknown')
# Fraud Indicator 1 for fraud found 0 for not found
data['Fraud_indicator'] = data['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Vehicle_Type'])


# Spliting the data into two indicator for dataset x and y that are derived the the main dataset FasTagfraudDetection.csv
X = data.drop(columns=['Fraud_indicator'])
# Indicator
y = data['Fraud_indicator']


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
# Using Attributes for preprocessing the pipelines
numeric_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Hour', 'Day_of_Week']
# Categorial Features
categorical_features = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Geographical_Location', 'Vehicle_Dimensions']

# Imputing Pipeline
numeric_transformer = Pipeline(steps=[
    # imputer
    ('imputer', SimpleImputer(strategy='median')),
    # Scaler
    ('scaler', StandardScaler())])

# Categorial Impute
categorical_transformer = Pipeline(steps=[
    # imputer
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    # oneshot
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# category and numeric preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        # Transformers
        ('num', numeric_transformer, numeric_features),
        # categorial
        ('cat', categorical_transformer, categorical_features)])

# Model pipeline
model = Pipeline(steps=[
    # Pipeline
    ('preprocessor', preprocessor),
    # classifier
    ('classifier', RandomForestClassifier(random_state=42))
])

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
# X training and testing data
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Preprocess the data
# the data is being divided into two parts X and Y. X will be used for traning the data and y will be used for testing.
X_train_processed = preprocessor.fit_transform(X_train_res)
# x testing
X_test_processed = preprocessor.transform(X_test)

# Train the model

model.fit(X_train_processed, y_train_res)

# Prediction on DataSet Y 
y_pred = model.predict(X_test_processed)

# Model Performance Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
# Classification
print('Classification Report:')
# Classification Report
print(classification_report(y_test, y_pred))
# Confusion Matrix
print('Confusion Matrix:')
# Y Test and Y Pred
print(confusion_matrix(y_test, y_pred))

# Feature importance
# the features that are used for predicting the fraud
feature_importance = model.named_steps['classifier'].feature_importances_
# Named Steps
feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
# ALL Features Names
all_feature_names = numeric_features + list(feature_names)
# Feature Importance
feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': feature_importance})
# Sort
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importance
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
# 10 Impoertant Feature 
plt.title('Top 10 Important Features')
# Model Show in bar
plt.show()
