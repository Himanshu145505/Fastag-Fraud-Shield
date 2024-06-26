# Required Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Mode for Selection
from sklearn.model_selection import train_test_split
# model for preprocessing the data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# model for composing
from sklearn.compose import ColumnTransformer
# for setting pipeline
from sklearn.pipeline import Pipeline
# for imputing
from sklearn.impute import SimpleImputer
# ensemle using Random Forest
from sklearn.ensemble import RandomForestClassifier
# Metrics of data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
# Flask Application Imports
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('FastagFraudDetection.csv')

# Data exploration
print(data.info())
print(data.describe())
print(data['Fraud_indicator'].value_counts())


sns.countplot(x='Fraud_indicator', data=data)
plt.show()

# Data preprocessing
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%m/%d/%Y %H:%M')
data['Hour'] = data['Timestamp'].dt.hour
data['Day_of_Week'] = data['Timestamp'].dt.dayofweek
data = data.drop(columns=['Transaction_ID', 'Timestamp', 'Vehicle_Plate_Number'])
data['FastagID'] = data['FastagID'].fillna('Unknown')
data['Fraud_indicator'] = data['Fraud_indicator'].map({'Fraud': 1, 'Not Fraud': 0})

X = data.drop(columns=['Fraud_indicator'])
y = data['Fraud_indicator']

# Split the dataset
# So we have one data file that contains data Fastag Fraud So we will split that data into two parts one will be used for traning and the second part will be used for testing the accuracy of predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
# Some of the Attributes for Pipeline
numeric_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Hour', 'Day_of_Week']
categorical_features = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Geographical_Location', 'Vehicle_Dimensions']

# Numeric Imputing the Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Categorical Imputing of Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Preprocessing the Pipleline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Preprocess the training data first
X stand for the first set of data that will be used for training and Y stand for 2nd set of data that will be used for testing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_transformed, y_train)

# Train the model
# Training the Model Based on the dataset 
model.named_steps['classifier'].fit(X_train_res, y_train_res)

# Predict on test data
# Using the second set of data for prediction
y_pred = model.named_steps['classifier'].predict(X_test_transformed)

# Evaluate the model
# Calculating the Prediction Accuracy (How much it Mathches with the result of training)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Feature importance
# The Attributes that are used for predictions 'Transaction_Amount', 'Amount_paid', 'Vehicle_Speed', 'Hour', 'Day_of_Week and other categorial information like geographical area, vechicle dimention etc.
feature_importance = model.named_steps['classifier'].feature_importances_
feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
all_feature_names = numeric_features + list(feature_names)
feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importance
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features Used For Prediction')
plt.show()

# Flask app for real-time prediction
app = Flask(__name__)

# Real time predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    # Preprocess and predict
    df_transformed = model.named_steps['preprocessor'].transform(df)
    prediction = model.named_steps['classifier'].predict(df_transformed)
    result = {'Fraud_indicator': int(prediction[0])}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
    
