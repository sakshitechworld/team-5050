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
from pre_processor.training_data_pre_processor import training_data_preprocessor, X_train, y_train
from pre_processor.prediction_data_pre_processor import to_predict_df

# Build a pipeline for the models
# Logistic Regression
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', LogisticRegression(max_iter=100))
])

# Fit the models
log_reg_pipeline.fit(X_train, y_train)

# Predict using the trained models
y_pred_log_reg = log_reg_pipeline.predict(to_predict_df)

print(len(y_pred_log_reg))
