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
# log_reg_pipeline = Pipeline(steps=[
#     ('preprocessor', training_data_preprocessor),
#     ('classifier', LogisticRegression(max_iter=100))
# ])

# # Decision Tree Classifier
# dt_pipeline = Pipeline(steps=[
#     ('preprocessor', training_data_preprocessor),
#     ('classifier', DecisionTreeClassifier(random_state=100))
# ])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(X_train)
print(y_train)
# Fit the models
# log_reg_pipeline.fit(X_train, y_train)
# dt_pipeline.fit(X_train, y_train)

# # Predict using the trained models
# y_pred_log_reg = log_reg_pipeline.predict(X_test)
# y_pred_dt = dt_pipeline.predict(X_test)


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the target variable
y_train_encoded = label_encoder.fit_transform(y_train)

# You can apply the same transformation to y_test as well
y_test_encoded = label_encoder.transform(y_test)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
# Define the categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)

leaderboard_test_data = pd.read_csv('D:/work/projects/repo/team-5050/data/test.csv')
# leaderboard_test_data = leaderboard_test_data.values


# Fit and transform the data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

import xgboost as xgb

# Train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train_encoded, y_train_encoded)

# Make predictions
y_pred = model.predict(X_test_encoded)


import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN


resampling_methods = {
    'RandomOverSampler': RandomOverSampler()
}

# Initialize the LabelEncoder for target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Preprocessing: Apply OneHotEncoder to categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown = "ignore"), categorical_cols)],
    remainder='passthrough'
)

# Apply the preprocessing transformation to X_train and X_test before resampling
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(leaderboard_test_data)

# Initialize the XGBoost model
model = xgb.XGBClassifier(random_state=42)

# Loop through resampling methods
for resample_name, resample_method in resampling_methods.items():
    print(f"Resampling Method: {resample_name}")
    
    # Apply resampling if not 'None'
    if resample_name != 'None':
        X_resampled, y_resampled = resample_method.fit_resample(X_train_encoded, y_train_encoded)
    else:
        X_resampled, y_resampled = X_train_encoded, y_train_encoded

   # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    # Make predictions
    y_pred = model.predict(X_test_encoded)

    # leaderboard_test_data = pd.DataFrame(leaderboard_test_data) 
    y_pred = pd.DataFrame(y_pred, columns=["y"])
    y_pred['y'].replace([0, 1], ["no", "yes"], inplace=True)
    print("printing y_pred", y_pred)

    leaderboard_check = pd.concat([leaderboard_test_data, y_pred], axis=1)

    leaderboard_check.to_csv('5050.csv', index=False)

#     # Calculate metrics
#     accuracy = accuracy_score(y_test_encoded, y_pred)
#     f1 = f1_score(y_test_encoded, y_pred)
#     roc_auc = roc_auc_score(y_test_encoded, model.predict_proba(X_test_encoded)[:, 1])
#     conf_matrix = confusion_matrix(y_test_encoded, y_pred)
#     class_report = classification_report(y_test_encoded, y_pred)

#     # Print results
#     print(f"Accuracy: {accuracy}")
#     print(f"F1 Score: {f1}")
#     print(f"ROC AUC: {roc_auc}")
#     print(f"Confusion Matrix:\n{conf_matrix}")
#     print(f"Classification Report:\n{class_report}")
#     print("-" * 50)


# # # Train and evaluate for each resampling method
# # for method_name, resampling_method in resampling_methods.items():
# #     train_and_evaluate(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, resampling_method)

# # # Also evaluate the model without resampling
# # train_and_evaluate(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, resampling_method=None)


# # # Evaluate the model
# # from sklearn.metrics import accuracy_score
# # print('Accuracy:', accuracy_score(y_test_encoded, y_pred))
# # print("Classification Report:")
# # print(classification_report(y_test_encoded, y_pred))

# # print("Confusion Matrix:")
# # cm = confusion_matrix(y_test_encoded, y_pred)
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
# # plt.xlabel('Predicted')
# # plt.ylabel('True')
# # plt.title('Confusion Matrix')
# # plt.show()

# # Calculate Precision, Recall, F1-Score
# precision = precision_score(y_test_encoded, y_pred)
# recall = recall_score(y_test_encoded, y_pred)
# f1 = f1_score(y_test_encoded, y_pred)

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")

# # Calculate AUC (Area Under the Curve)
# y_prob = model.predict_proba(X_test_encoded)[:, 1]  # Probabilities for class 1
# auc = roc_auc_score(y_test_encoded, y_prob)
# print(f"AUC: {auc}")

# # from sklearn.metrics import accuracy_score

# # # Calculate accuracy for Logistic Regression
# # log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
# # print("Logistic Regression Accuracy:", log_reg_accuracy)

# # # Calculate accuracy for Decision Tree
# # dt_accuracy = accuracy_score(y_test, y_pred_dt)
# # print("Decision Tree Accuracy:", dt_accuracy)

# # from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

# # # Classification Report
# # print("Logistic Regression Report:")
# # print(classification_report(y_test, y_pred_log_reg))
# # print("Decision Tree Report:")
# # print(classification_report(y_test, y_pred_dt))


# # # ROC-AUC Scores
# # log_reg_auc = roc_auc_score(y_test, log_reg_pipeline.predict_proba(X_test)[:, 1])
# # dt_auc = roc_auc_score(y_test, dt_pipeline.predict_proba(X_test)[:, 1])
# # print(f"Logistic Regression AUC: {log_reg_auc}")
# # print(f"Decision Tree AUC: {dt_auc}")


# # # Confusion Matrices
# # print("Logistic Regression Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred_log_reg))
# # print("Decision Tree Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred_dt))


# # # F1 Scores
# # log_reg_f1 = f1_score(y_test, y_pred_log_reg)
# # dt_f1 = f1_score(y_test, y_pred_dt)
# # print(f"Logistic Regression F1 Score: {log_reg_f1}")
# # print(f"Decision Tree F1 Score: {dt_f1}")



# # #import numpy as np
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.svm import SVC
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.neighbors import KNeighborsClassifier
# # import xgboost as xgb
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
# # from sklearn.impute import SimpleImputer
# # from pre_processor.training_data_pre_processor import training_data_preprocessor, X_test, X_train, y_train, y_test

# # # Support Vector Machine (SVM) Model
# # svm_pipeline = Pipeline(steps=[
# #     ('preprocessor', training_data_preprocessor),
# #     ('classifier', SVC(probability=True, random_state=42))
# # ])

# # # Random Forest Model
# # rf_pipeline = Pipeline(steps=[
# #     ('preprocessor', training_data_preprocessor),
# #     ('classifier', RandomForestClassifier(random_state=42))
# # ])

# # # K-Nearest Neighbors (KNN) Model
# # knn_pipeline = Pipeline(steps=[
# #     ('preprocessor', training_data_preprocessor),
# #     ('classifier', KNeighborsClassifier())
# # ])

# # # XGBoost Model
# # xgb_pipeline = Pipeline(steps=[
# #     ('preprocessor', training_data_preprocessor),
# #     ('classifier', xgb.XGBClassifier(random_state=42))
# # ])

# # # List of all models to iterate through
# # pipelines = {
# #     'Logistic Regression': log_reg_pipeline,
# #     'Decision Tree': dt_pipeline,
# #     'SVM': svm_pipeline,
# #     'Random Forest': rf_pipeline,
# #     'KNN': knn_pipeline,
# #     'XGBoost': xgb_pipeline
# # }

# # # Train, predict, and evaluate each model
# # for model_name, pipeline in pipelines.items():
# #     print(f"Training {model_name}...")
    
# #     # Train the model
# #     pipeline.fit(X_train, y_train)
    

# #     # Make predictions
# #     y_pred = pipeline.predict(X_test)
# #     y_pred_xgb = xgb_model.predict(X_test)

# #     # Accuracy
# #     accuracy = accuracy_score(y_test, y_pred)
# #     print(f"{model_name} Accuracy: {accuracy}")
    
# #     # Classification report
# #     print(f"{model_name} Classification Report:")
# #     print(classification_report(y_test, y_pred))
    
# #     # ROC-AUC
# #     auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
# #     print(f"{model_name} AUC: {auc}")
    
# #     # Confusion matrix
# #     print(f"{model_name} Confusion Matrix:")
# #     print(confusion_matrix(y_test, y_pred))
    
# #     # F1 score
# #     f1 = f1_score(y_test, y_pred)
# #     print(f"{model_name} F1 Score: {f1}")
    
# #     print("-" * 50)





# # import numpy as np
# # import pandas as pd
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
# # from pre_processor.training_data_pre_processor import training_data_preprocessor, X_test, X_train, y_train, y_test

# # # Define the parameter grid for Logistic Regression
# # log_reg_param_grid = {
# #     'classifier__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
# #     'classifier__solver': ['liblinear', 'saga'],  # Solver options
# #     'classifier__class_weight': ['balanced', None],  # Handling imbalanced classes
# #     'classifier__max_iter': [100, 200, 300]  # Iterations for solver convergence
# # }

# # # Define the parameter grid for Decision Tree
# # dt_param_grid = {
# #     'classifier__max_depth': [3, 5, 7, 10, None],  # Maximum depth of the tree
# #     'classifier__min_samples_split': [2, 5, 10],  # Minimum samples to split a node
# #     'classifier__min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf
# #     'classifier__class_weight': ['balanced', None],  # Handling imbalanced classes
# #     'classifier__max_features': ['auto', 'sqrt', 'log2', None],  # Feature selection
# # }

# # # Create the Logistic Regression pipeline
# # log_reg_pipeline = Pipeline(steps=[
# #     ('preprocessor', training_data_preprocessor),
# #     ('classifier', LogisticRegression())
# # ])

# # # Create the Decision Tree pipeline
# # dt_pipeline = Pipeline(steps=[
# #     ('preprocessor', training_data_preprocessor),
# #     ('classifier', DecisionTreeClassifier(random_state=42))
# # ])

# # # Split the data for training
# # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # # Perform Grid Search for Logistic Regression
# # log_reg_grid_search = GridSearchCV(estimator=log_reg_pipeline, param_grid=log_reg_param_grid, 
# #                                    cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# # # Fit the model with GridSearchCV
# # log_reg_grid_search.fit(X_train, y_train)

# # # Get best parameters and best score for Logistic Regression
# # print("Best Logistic Regression Parameters:", log_reg_grid_search.best_params_)
# # print("Best Logistic Regression Score:", log_reg_grid_search.best_score_)

# # # Predict using the best Logistic Regression model
# # y_pred_log_reg = log_reg_grid_search.predict(X_test)

# # # Calculate accuracy for Logistic Regression
# # log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
# # print("Logistic Regression Accuracy:", log_reg_accuracy)

# # # Classification Report for Logistic Regression
# # print("\nLogistic Regression Report:")
# # print(classification_report(y_test, y_pred_log_reg))

# # # ROC-AUC Score for Logistic Regression
# # log_reg_auc = roc_auc_score(y_test, log_reg_grid_search.predict_proba(X_test)[:, 1])
# # print(f"Logistic Regression AUC: {log_reg_auc}")

# # # Confusion Matrix for Logistic Regression
# # print("Logistic Regression Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred_log_reg))

# # # F1 Score for Logistic Regression
# # log_reg_f1 = f1_score(y_test, y_pred_log_reg)
# # print(f"Logistic Regression F1 Score: {log_reg_f1}")

# # # Perform Grid Search for Decision Tree
# # dt_grid_search = GridSearchCV(estimator=dt_pipeline, param_grid=dt_param_grid, 
# #                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# # # Fit the model with GridSearchCV
# # dt_grid_search.fit(X_train, y_train)

# # # Get best parameters and best score for Decision Tree
# # print("\nBest Decision Tree Parameters:", dt_grid_search.best_params_)
# # print("Best Decision Tree Score:", dt_grid_search.best_score_)

# # # Predict using the best Decision Tree model
# # y_pred_dt = dt_grid_search.predict(X_test)

# # # Calculate accuracy for Decision Tree
# # dt_accuracy = accuracy_score(y_test, y_pred_dt)
# # print("Decision Tree Accuracy:", dt_accuracy)

# # # Classification Report for Decision Tree
# # print("\nDecision Tree Report:")
# # print(classification_report(y_test, y_pred_dt))

# # # ROC-AUC Score for Decision Tree
# # dt_auc = roc_auc_score(y_test, dt_grid_search.predict_proba(X_test)[:, 1])
# # print(f"Decision Tree AUC: {dt_auc}")

# # # Confusion Matrix for Decision Tree
# # print("Decision Tree Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred_dt))

# # # F1 Score for Decision Tree
# # dt_f1 = f1_score(y_test, y_pred_dt)
# # print(f"Decision Tree F1 Score: {dt_f1}")








