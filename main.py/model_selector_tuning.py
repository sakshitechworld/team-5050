import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Load dataset
data = pd.read_csv('D:/work/projects/repo/team-5050/data/train.csv')

# Feature and target separation
X = data[['age', 'balance', 'default', 'day', 'housing', 'duration', 'loan', 'campaign', 
          'pdays', 'previous', 'job', 'marital', 'education', 'contact', 'month', 'poutcome']]
y = LabelEncoder().fit_transform(data['y'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify columns
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_columns = [col for col in X_train.columns if col not in numerical_columns]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Resampling methods
resampling_methods = {
    'None': None,
    'SMOTE': SMOTE(random_state=42),
    'RandomOverSampler': RandomOverSampler(random_state=42),
    'RandomUnderSampler': RandomUnderSampler(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

# Evaluate each model with different resampling methods
for resample_name, resample_method in resampling_methods.items():
    print(f"\n--- Resampling Method: {resample_name} ---")
    
    if resample_method is not None:
        # If resampling, use ImbPipeline
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('resampler', resample_method)
        ])
        X_train_transformed, y_train_resampled = pipeline.fit_resample(X_train, y_train)
    else:
        # If no resampling, use standard Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])
        X_train_transformed = pipeline.fit_transform(X_train)
        y_train_resampled = y_train  # No resampling
    
    # Preprocess the test data using only the preprocessor
    X_test_transformed = preprocessor.transform(X_test)
    
    # Evaluate models
    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")
        
        # Train the model
        model.fit(X_train_transformed, y_train_resampled)
        
        # Predictions
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Print results
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    ROC AUC: {roc_auc}")
        print(f"    Confusion Matrix:\n{conf_matrix}")


# Generate predictions on the test data
y_pred = model.predict(X_test_transformed)

# Prepare the DataFrame with the required columns (index as ID and predictions)
prediction_df = pd.DataFrame({
    'ID': X_test.index,  # Use the index of the test data as ID
    'y_pred': y_pred
})

# Save the predictions to a CSV file
prediction_df.to_csv('5050.csv', index=False)

