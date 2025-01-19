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


from sklearn.metrics import accuracy_score

# Calculate accuracy for Logistic Regression
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", log_reg_accuracy)

# Calculate accuracy for Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_accuracy)

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

# Classification Report
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Decision Tree Report:")
print(classification_report(y_test, y_pred_dt))


# ROC-AUC Scores
log_reg_auc = roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test)[:, 1])
dt_auc = roc_auc_score(y_test, dt_pipeline.predict_proba(X_test)[:, 1])
print(f"Logistic Regression AUC: {log_reg_auc}")
print(f"Decision Tree AUC: {dt_auc}")


# Confusion Matrices
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))


# F1 Scores
log_reg_f1 = f1_score(y_test, y_pred_log_reg)
dt_f1 = f1_score(y_test, y_pred_dt)
print(f"Logistic Regression F1 Score: {log_reg_f1}")
print(f"Decision Tree F1 Score: {dt_f1}")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from pre_processor.training_data_pre_processor import training_data_preprocessor, X_test, X_train, y_train, y_test

# Support Vector Machine (SVM) Model
svm_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

# Random Forest Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# K-Nearest Neighbors (KNN) Model
knn_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', KNeighborsClassifier())
])


# List of all models to iterate through
pipelines = {
    'Logistic Regression': log_reg_pipeline,
    'Decision Tree': dt_pipeline,
    'SVM': svm_pipeline,
    'Random Forest': rf_pipeline,
    'KNN': knn_pipeline
}

# Train, predict, and evaluate each model
for model_name, pipeline in pipelines.items():
    print(f"Training {model_name}...")
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    
    # Classification report
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    print(f"{model_name} AUC: {auc}")
    
    # Confusion matrix
    print(f"{model_name} Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # F1 score
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} F1 Score: {f1}")
    
    print("-" * 50)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from pre_processor.training_data_pre_processor import training_data_preprocessor, X_test, X_train, y_train, y_test

# Define the parameter grid for Logistic Regression
log_reg_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'classifier__solver': ['liblinear', 'saga'],  # Solver options
    'classifier__class_weight': ['balanced', None],  # Handling imbalanced classes
    'classifier__max_iter': [100, 200, 300]  # Iterations for solver convergence
}

# Define the parameter grid for Decision Tree
dt_param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf
    'classifier__class_weight': ['balanced', None],  # Handling imbalanced classes
    'classifier__max_features': ['auto', 'sqrt', 'log2', None],  # Feature selection
}

# Create the Logistic Regression pipeline
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', LogisticRegression())
])

# Create the Decision Tree pipeline
dt_pipeline = Pipeline(steps=[
    ('preprocessor', training_data_preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Split the data for training
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Perform Grid Search for Logistic Regression
log_reg_grid_search = GridSearchCV(estimator=log_reg_pipeline, param_grid=log_reg_param_grid, 
                                   cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit the model with GridSearchCV
log_reg_grid_search.fit(X_train, y_train)

# Get best parameters and best score for Logistic Regression
print("Best Logistic Regression Parameters:", log_reg_grid_search.best_params_)
print("Best Logistic Regression Score:", log_reg_grid_search.best_score_)

# Predict using the best Logistic Regression model
y_pred_log_reg = log_reg_grid_search.predict(X_test)

# Calculate accuracy for Logistic Regression
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", log_reg_accuracy)

# Classification Report for Logistic Regression
print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred_log_reg))

# ROC-AUC Score for Logistic Regression
log_reg_auc = roc_auc_score(y_test, log_reg_grid_search.predict_proba(X_test)[:, 1])
print(f"Logistic Regression AUC: {log_reg_auc}")

# Confusion Matrix for Logistic Regression
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

# F1 Score for Logistic Regression
log_reg_f1 = f1_score(y_test, y_pred_log_reg)
print(f"Logistic Regression F1 Score: {log_reg_f1}")

# Perform Grid Search for Decision Tree
dt_grid_search = GridSearchCV(estimator=dt_pipeline, param_grid=dt_param_grid, 
                              cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit the model with GridSearchCV
dt_grid_search.fit(X_train, y_train)

# Get best parameters and best score for Decision Tree
print("\nBest Decision Tree Parameters:", dt_grid_search.best_params_)
print("Best Decision Tree Score:", dt_grid_search.best_score_)

# Predict using the best Decision Tree model
y_pred_dt = dt_grid_search.predict(X_test)

# Calculate accuracy for Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_accuracy)

# Classification Report for Decision Tree
print("\nDecision Tree Report:")
print(classification_report(y_test, y_pred_dt))

# ROC-AUC Score for Decision Tree
dt_auc = roc_auc_score(y_test, dt_grid_search.predict_proba(X_test)[:, 1])
print(f"Decision Tree AUC: {dt_auc}")

# Confusion Matrix for Decision Tree
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# F1 Score for Decision Tree
dt_f1 = f1_score(y_test, y_pred_dt)
print(f"Decision Tree F1 Score: {dt_f1}")



# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTEENN

# # Define the resampling strategies
# smote = SMOTE(random_state=42)
# under_sampler = RandomUnderSampler(random_state=42)
# smote_enn = SMOTEENN(random_state=42)

# # Apply SMOTE to oversample the minority class
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# print("After SMOTE:", X_train_smote.shape, y_train_smote.value_counts())

# # Apply RandomUnderSampler to undersample the majority class
# X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
# print("After Undersampling:", X_train_under.shape, y_train_under.value_counts())

# # Apply SMOTEENN to combine oversampling and undersampling
# X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)
# print("After SMOTEENN:", X_train_smoteenn.shape, y_train_smoteenn.value_counts())


# # Logistic Regression on SMOTE data
# log_reg_pipeline.fit(X_train_smote, y_train_smote)
# y_pred_log_reg_smote = log_reg_pipeline.predict(X_test)
# log_reg_auc_smote = roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test)[:, 1])
# print("Logistic Regression AUC (SMOTE):", log_reg_auc_smote)
# print(classification_report(y_test, y_pred_log_reg_smote))

# # Decision Tree on SMOTE data
# dt_pipeline.fit(X_train_smote, y_train_smote)
# y_pred_dt_smote = dt_pipeline.predict(X_test)
# dt_auc_smote = roc_auc_score(y_test, dt_pipeline.predict_proba(X_test)[:, 1])
# print("Decision Tree AUC (SMOTE):", dt_auc_smote)
# print(classification_report(y_test, y_pred_dt_smote))




# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
# from imblearn.over_sampling import SMOTE

# Assuming X_train and y_train are already defined

# Step 1: Verify all columns are numeric
# print("Data types of X_train before encoding:")
# print(X_train.dtypes)

# If there are non-numeric columns, encode them
# (Skip this if preprocessing was already done)
# categorical_cols = X_train.select_dtypes(include=['object']).columns
# if not categorical_cols.empty:
#     encoder = OneHotEncoder(drop='first', sparse=False)
#     encoded_data = encoder.fit_transform(X_train[categorical_cols])
#     encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
#     X_train = pd.concat([X_train.drop(columns=categorical_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# print("Data types of X_train after encoding:")
# print(X_train.dtypes)

# # Step 2: Apply SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Step 3: Resample X_test if needed
# # Ensure X_test matches the format of X_train (e.g., encoding)

# # Step 4: Refit models with resampled data
# log_reg_pipeline.fit(X_train_resampled, y_train_resampled)
# dt_pipeline.fit(X_train_resampled, y_train_resampled)

# # Step 5: Predict on test data
# y_pred_log_reg = log_reg_pipeline.predict(X_test)
# y_pred_dt = dt_pipeline.predict(X_test)

# # Step 6: Evaluate metrics
# print("Logistic Regression Report:")
# print(classification_report(y_test, y_pred_log_reg))
# print("Decision Tree Report:")
# print(classification_report(y_test, y_pred_dt))

# print("Logistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_log_reg))
# print("Decision Tree Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_dt))

# log_reg_auc = roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test)[:, 1])
# dt_auc = roc_auc_score(y_test, dt_pipeline.predict_proba(X_test)[:, 1])
# print(f"Logistic Regression AUC: {log_reg_auc}")
# print(f"Decision Tree AUC: {dt_auc}")

# log_reg_f1 = f1_score(y_test, y_pred_log_reg)
# dt_f1 = f1_score(y_test, y_pred_dt)
# print(f"Logistic Regression F1 Score: {log_reg_f1}")
# print(f"Decision Tree F1 Score: {dt_f1}")



# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTEENN

# # Define the resampling strategies
# smote = SMOTE(random_state=42)
# under_sampler = RandomUnderSampler(random_state=42)
# smote_enn = SMOTEENN(random_state=42)

# # Apply SMOTE to oversample the minority class
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# print("After SMOTE:", X_train_smote.shape, y_train_smote.value_counts())

# # Apply RandomUnderSampler to undersample the majority class
# X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
# print("After Undersampling:", X_train_under.shape, y_train_under.value_counts())

# # Apply SMOTEENN to combine oversampling and undersampling
# X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train, y_train)
# print("After SMOTEENN:", X_train_smoteenn.shape, y_train_smoteenn.value_counts())

# # Convert resampled data back to DataFrame
# # Check if X_train is a DataFrame, else provide columns from the preprocessor
# if isinstance(X_train, pd.DataFrame):
#     columns = X_train.columns
# else:
#     columns = [f"feature_{i}" for i in range(X_train.shape[1])]

# X_train_smote = pd.DataFrame(X_train_smote, columns=columns)
# X_train_under = pd.DataFrame(X_train_under, columns=columns)
# X_train_smoteenn = pd.DataFrame(X_train_smoteenn, columns=columns)

# # Fit the models using the resampled data
# log_reg_pipeline.fit(X_train_smote, y_train_smote)
# dt_pipeline.fit(X_train_smote, y_train_smote)

# # Predict using the trained models
# y_pred_log_reg_smote = log_reg_pipeline.predict(X_test)
# y_pred_dt_smote = dt_pipeline.predict(X_test)

# # Evaluate Metrics
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# # Logistic Regression and Decision Tree Classification Reports
# print("\nLogistic Regression Report (SMOTE):")
# print(classification_report(y_test, y_pred_log_reg_smote))
# print("Decision Tree Report (SMOTE):")
# print(classification_report(y_test, y_pred_dt_smote))

# # Confusion Matrices
# print("\nLogistic Regression Confusion Matrix (SMOTE):")
# print(confusion_matrix(y_test, y_pred_log_reg_smote))
# print("Decision Tree Confusion Matrix (SMOTE):")
# print(confusion_matrix(y_test, y_pred_dt_smote))

# # AUC Scores
# log_reg_auc_smote = roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test)[:, 1])
# dt_auc_smote = roc_auc_score(y_test, dt_pipeline.predict_proba(X_test)[:, 1])
# print(f"Logistic Regression AUC (SMOTE): {log_reg_auc_smote}")
# print(f"Decision Tree AUC (SMOTE): {dt_auc_smote}")

# # F1 Scores
# log_reg_f1_smote = f1_score(y_test, y_pred_log_reg_smote)
# dt_f1_smote = f1_score(y_test, y_pred_dt_smote)
# print(f"Logistic Regression F1 Score (SMOTE): {log_reg_f1_smote}")
# print(f"Decision Tree F1 Score (SMOTE): {dt_f1_smote}")


