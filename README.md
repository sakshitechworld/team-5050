##Bank Term Deposit Prediction - Team_participants_5050

###Problem Statement: 
---This project aims to optimize telephonic marketing campaigns for a Portuguese bank by predicting whether a client will subscribe to a term deposit (y). The dataset includes client ------details, interaction history, and campaign outcomes, enabling the creation of a predictive classification model.

###Dataset Overview
---The datasets used in this project contain client information, interaction history, and campaign outcomes:
---Training Dataset (train.csv): 40,000 rows and 17 columns.
---Test Dataset (test.csv): 5,211 rows and 16 columns (without the target variable).

###Dataset Features:
---a) client information
---b) Campaign-Related Features:
---c) Target variables

###Project Features
---Data Preprocessing:
---Handle missing values.
---Encode categorical variables.
---Scale numeric data for optimal model performance.
---Machine Learning Models:
---Build predictive models for binary classification of the target variable y.
---Model Evaluation:
---Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.

###Project Structure
---team-5050-main/
---├── README.md             # Project documentation
---├── data/
---│   ├── train.csv         # Training dataset
---│   └── test.csv          # Test dataset
---├── main.py               # Script for data preprocessing and model training
---├── requirements.txt      # Dependencies for the project
---└── LICENSE               # License information


###Prerequisites
---Python 3.8 or later
---Required Python libraries listed in requirements.txt

###Setup Instructions
---git clone [repository_url]
---cd team-5050-main
---python3 -m venv venv
---source venv/bin/activate  # For Windows: venv\Scripts\activate
---pip install -r requirements.txt
---python main.py


###Expected Results
---The predictive model will classify whether a client will subscribe to a term deposit.
---Evaluation metrics such as accuracy, precision, recall, and F1-score will be used to assess the model's performance.

