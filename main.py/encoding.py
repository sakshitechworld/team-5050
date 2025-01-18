import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('D:/work/projects/repo/team-5050/data/train.csv')

# Label Encoding for binary columns
label_encoder = LabelEncoder()
data['default'] = label_encoder.fit_transform(data['default'])
data['housing'] = label_encoder.fit_transform(data['housing'])
data['loan'] = label_encoder.fit_transform(data['loan'])
data['y'] = label_encoder.fit_transform(data['y'])

# One-Hot Encoding for multi-class categorical columns (e.g., job, marital)
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], drop_first=True)

print(data.head())
print(data.info())  # Confirm the new columns and their data types

print(data[['job_blue-collar', 'job_entrepreneur']].head())

# Add a flag for high-contact cases in the 'campaign' column
data['high_campaign_flag'] = (data['campaign'] > 6).astype(int)
# Verify the new column
print(data[['campaign', 'high_campaign_flag']].head())

#Split Data into Features and Target: Separate the input features (X) and target variable (y) for modeling
X = data.drop('y', axis=1)  # Drop the target column from features
y = data['y']                # Target column

#Train-Test Split: Divide the data into training and testing sets to evaluate your model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])


correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

feature_importances = pd.DataFrame({'Feature': X_train.columns, 
                                    'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
print(feature_importances.head(10))


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Remove low-importance features
low_importance_features = ['pdays']  # Add more features based on correlation and importance analysis
X_train = X_train.drop(columns=low_importance_features)
X_test = X_test.drop(columns=low_importance_features)

# Step 2: Create interaction terms
X_train['duration_balance'] = X_train['duration'] * X_train['balance']
X_train['age_balance'] = X_train['age'] * X_train['balance']
X_test['duration_balance'] = X_test['duration'] * X_test['balance']
X_test['age_balance'] = X_test['age'] * X_test['balance']

# Step 3: Train a Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Grid Search for hyperparameter tuning
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Best Parameters:\n", grid_search.best_params_)
