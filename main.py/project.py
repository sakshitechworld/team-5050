import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('D:/work/projects/repo/team-5050/data/train.csv')

# Display the first few rows of the data
print(data.head())

# Check the shape of the dataset (rows, columns)
print(data.shape)

# Get summary statistics for numerical columns
print(data.describe())

# Check for missing values in the dataset
print(data.isnull().sum())

# Get information about the data types and non-null counts
print(data.info())

# Check unique values in categorical columns (e.g., job, marital)
print(data['job'].unique())
print(data['marital'].unique())

print(data['job'].value_counts())
print(data['marital'].value_counts())


# List of numerical columns
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Function to detect outliers using IQR
def detect_outliers_iqr(df, columns):
    outlier_info = {}
    for column in columns:
        Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
        Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_info[column] = {
            "num_outliers": len(outliers),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        print(f"Column: {column}")
        print(f"  Number of outliers: {len(outliers)}")
        print(f"  Lower bound: {lower_bound}")
        print(f"  Upper bound: {upper_bound}")
    return outlier_info

# Detect outliers and print summary
outlier_summary = detect_outliers_iqr(data, numerical_columns)

# Visualize outliers using box plots
for column in numerical_columns:
    plt.figure(figsize=(6, 4))
    data.boxplot(column=column)
    plt.title(f'Box Plot for {column}')
    plt.show()

# To see how outliers affect data distribution and relationships

# Histograms
for column in ['age', 'balance', 'duration', 'campaign']:
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Scatter Plot for balance vs. duration
sns.scatterplot(x=data['balance'], y=data['duration'])
plt.title('Balance vs. Duration')
plt.show()

print(data.columns)




