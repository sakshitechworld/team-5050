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

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

