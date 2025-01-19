## **Overview of the Project**
This project aims to predict whether a bank customer will subscribe to a term deposit based on demographic, financial, and campaign-related features. The dataset originates from a Portuguese banking institution’s direct marketing campaigns. Accurate predictions can help optimize telemarketing efforts and reduce costs.

## **Key Features and Technologies Used**

### **Key Features:**
- Prediction of customer subscription likelihood to term deposits.
- Insights into the effectiveness of telemarketing campaigns.
- Identification of high-priority customer segments for targeted outreach.

### **Technologies Used:**
- **Programming:** Python (pandas, seaborn, scikit-learn, matplotlib).
- **Visualization:** Power BI for interactive dashboards.
- **Machine Learning Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Tools:** Python libraries (xgboost, lightgbm), Power BI for visualizations.

## **Clear Setup Instructions**

1. **Set Up the Environment:**
   - Ensure Python is installed on your system.
   - Install the required dependencies by running:
     ```bash
     pip install pandas numpy scikit-learn seaborn matplotlib xgboost lightgbm
     ```

2. **Run EDA:**
   - Execute the script `project.py` to explore the data and visualize key patterns.
   - Example command:
     ```bash
     python project.py
     ```

3. **Preprocess Data:**
   - Use the `encoding.py` script for encoding features, scaling data, and engineering new features.
   - Example command:
     ```bash
     python encoding.py
     ```

4. **Train Models:**
   - Execute the `testing.py` script to train machine learning models and evaluate their performance.
   - Example command:
     ```bash
     python testing.py
     ```

5. **Generate Predictions:**
   - Use the best-performing model to predict on the test dataset.
   - Save the output in the format `5050.csv` by extending or modifying the existing scripts.
