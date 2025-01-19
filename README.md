## **Overview of the Project**
This project aims to predict whether a bank customer will subscribe to a term deposit based on demographic, financial, and campaign-related features. The dataset originates from a Portuguese banking institution’s direct marketing campaigns. Accurate predictions can help optimize telemarketing efforts and reduce costs.

### **Key Features and Technologies Used**

- **XGBoost Model**: A powerful and efficient gradient boosting algorithm used to train the model.
- **Random Over Sampler**: A resampling technique to handle class imbalance by oversampling the minority class in the training data.
- **Pandas**: Used for data manipulation and analysis.
- **Scikit-learn**: Provides tools for splitting the dataset, evaluating model performance, and applying machine learning metrics.
- **Matplotlib & Seaborn**: For visualization, including heatmaps and correlation plots.
- **Imbalanced-learn**: For handling class imbalance using resampling techniques.
- **F1 Score**: The model achieved an **F1 score of 0.55**, the highest among all models tested.

### **Presentation for BankReach Flow Project**
https://docs.google.com/presentation/d/1ziLrbxcoIEH_GO4t9kd1REQDeRcdzwhS/edit?usp=sharing&ouid=104123228965815355373&rtpof=true&sd=true

### **Walkthrough of the project**
https://drive.google.com/file/d/1s9Aby11B554VDEvRYa2u3EwoknA7_1Sg/view?usp=sharing

### **Dashboard**
https://drive.google.com/file/d/1W5mFNJlHcnUsChicmHWnLE81zSQV2ZBD/view?usp=sharing

## **Setup Instructions**

To set up and run this project, follow these steps:

### 1. Clone the Repository
Clone the project repository to your local machine using the following command:
```bash
git clone <repository-url>
cd <project-directory>

### 2. Install Dependencies**
Ensure that you have Python 3.x installed on your system. Use `pip` or `conda` to install the required libraries. Run the following command to install all dependencies:

```bash
pip install -r requirements.txt

##**Project Features**

-Data Preprocessing:
-Handle missing values.
-Encode categorical variables.
-Scale numeric data for optimal model performance.
-Machine Learning Models:
-Build predictive models for binary classification of the target variable y.
-Model Evaluation:
-Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.

##**Project Structure**

-team-5050-main/
-├── README.md             # Project documentation
-├── data/
-│   ├── train.csv         # Training dataset
-│   └── test.csv          # Test dataset
-├── main.py               # Script for data preprocessing and model training
-├── requirements.txt      # Dependencies for the project

---

##**Prerequisites**
-Python 3.8 or later
-Required Python libraries listed in requirements.txt

---

##**Setup Instructions**
-git clone [repository_url]
-cd team-5050-main
-python3 -m venv venv
-source venv/bin/activate  # For Windows: venv\Scripts\activate
-pip install -r requirements.txt
-python main.py

---

##**Expected Results**
-The predictive model will classify whether a client will subscribe to a term deposit.
-Evaluation metrics such as accuracy, precision, recall, and F1-score will be used to assess the model's performance.

5. **Generate Predictions:**
   - Use the best-performing model to predict on the test dataset.
   - Save the output in the format `5050.csv` by extending or modifying the existing scripts.
