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

# Sample dataset - Using a synthetic dataset as an example
from sklearn.preprocessing import LabelEncoder

# Load dataset
training_data = pd.read_csv('D:/work/projects/repo/team-5050/data/train.csv')
X = pd.DataFrame(training_data, columns= ['age', 'balance', 'default', 'day','housing', 'duration', 'loan', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education', 'contact', 'month', 'poutcome'])
y = LabelEncoder().fit_transform(training_data['y'])


# Add a synthetic categorical column for demonstration (if you don't have one in your dataset)
# For example: add a column with 'Yes' or 'No' values
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# 1. **Handle Categorical Columns** (using OneHotEncoder)
categorical_columns = [col for col in X.columns if col not in numerical_columns]

# 2. **Outlier Detection and Removal** using IQR
# def remove_outliers(df):
#     Q1 = df.quantile(0.25)
#     Q3 = df.quantile(0.75)
#     IQR = Q3 - Q1
#     df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
#     return df_no_outliers

# # Remove outliers from the numerical columns
# X_numerical_no_outliers = remove_outliers(X[numerical_columns])

# # Re-assign y values that correspond to the rows after outlier removal
# X_filtered = X_numerical_no_outliers 
# y_filtered = y[X_filtered.index]

# 3. **Normalization and Categorical Encoding** using ColumnTransformer and Pipeline
# We will apply StandardScaler to numerical features and OneHotEncoder to categorical features.

training_data_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns), 
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# 4. **Train-Test Split** (Split data into training and testing sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



