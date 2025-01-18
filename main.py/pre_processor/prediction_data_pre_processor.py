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
data = pd.read_csv('D:/work/projects/repo/team-5050/data/test.csv')
to_predict_df = pd.DataFrame(data, columns= ['age', 'balance', 'default', 'day','housing', 'duration', 'loan', 'campaign', 'pdays', 'previous', 'job', 'marital', 'education', 'contact', 'month', 'poutcome'])


