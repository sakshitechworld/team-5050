import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from math import sqrt
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from pre_processor.training_data_pre_processor import training_data_preprocessor, X_test, X_train, y_train, y_test

# Build a pipeline for the models
# Logistic Regression
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', LogisticRegression(max_iter=100))
])

# Decision Tree Classifier
dt_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=100))
])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(X_train)
print(y_train)
# Fit the models
log_reg_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)

# Predict using the trained models
y_pred_log_reg = log_reg_pipeline.predict(X_test)
y_pred_dt = dt_pipeline.predict(X_test)

# Evaluate the models using RMSE
log_reg_rmse = sqrt(mean_squared_error(y_test, y_pred_log_reg))
dt_rmse = sqrt(mean_squared_error(y_test, y_pred_dt))

print(f"Logistic Regression RMSE: {log_reg_rmse}")
print(f"Decision Tree RMSE: {dt_rmse}")
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))


