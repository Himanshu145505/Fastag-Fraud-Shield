# Required Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing, pipeline, compose, ensemble , impute and other imports from Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
print(data.describe())
print(data['Fraud_indicator'].value_counts())

sns.countplot(x='Fraud_indicator', data=data)
plt.show()

# Data preprocessing
# Preprocessing Fastag Id, Transaction Id and other details
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%d/%Y %H:%M')
data['Hour'] = data['Timestamp'].dt.hour
data['Day_of_Week'] = data['Timestamp'].dt.dayofweek
data = data.drop(columns=['Transaction_ID', 'Timestamp', 'Vehicle_Plate_Number'])
data['FastagID'] = data['FastagID'].fillna('Unknown')
data['Fraud_indicator'] = data['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Vehicle_Type'])


# Spliting the data into two indicator for dataset x and y that are derived the the main dataset FasTagfraudDetection.csv
X = data.drop(columns=['Fraud_indicator'])
y = data['Fraud_indicator']


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
# Using Attributes for preprocessing the pipelines
numeric_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Hour', 'Day_of_Week']
categorical_features = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Geographical_Location', 'Vehicle_Dimensions']

# Imputing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# category and numeric preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Preprocess the data
# the data is being divided into two parts X and Y. X will be used for traning the data and y will be used for testing.
X_train_processed = preprocessor.fit_transform(X_train_res)
X_test_processed = preprocessor.transform(X_test)

# Train the model

model.fit(X_train_processed, y_train_res)

# Prediction on DataSet Y 
y_pred = model.predict(X_test_processed)

# Model Performance Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Feature importance
# the features that are used for predicting the fraud
feature_importance = model.named_steps['classifier'].feature_importances_
feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
all_feature_names = numeric_features + list(feature_names)
feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importance
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features')
plt.show()
