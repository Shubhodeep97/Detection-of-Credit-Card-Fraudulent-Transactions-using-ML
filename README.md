# Detection of Credit Card Fraud Transactions using Machine Learning

## Introduction

The main objective of this project is to detect anomalies and assist credit card organizations in taking relevant actions to prevent malicious activities related to fraudulent credit card transactions. It is crucial for credit card companies to recognize fraudulent transactions to ensure that customers are not charged for items they did not purchase.

The project encompasses end-to-end data analysis of transaction data to train a machine learning model capable of accurately predicting whether a transaction is fraudulent or normal. This involves data extraction, cleaning, and analysis to generate insights, followed by the development and training of a predictive model to enhance future fraud detection capabilities.


![Macro-image-of-all-major-credit-card-companies](https://github.com/user-attachments/assets/5a42ea24-29fb-4601-a859-0c213103e051)

![Machine_Learning](https://github.com/user-attachments/assets/aabdc2de-4257-4b52-a522-5d6fbedd8faa)
## Approach
The project involves the following steps:

- **Data Collection** : Gathering a comprehensive dataset of credit card transactions, including both legitimate and fraudulent ones.
- **Data Preprocessing** : Cleaning the data, handling missing values, and addressing the imbalanced nature of the classes to ensure consistency and accuracy in the model's inputs.
- **Exploratory Data Analysis (EDA)** : Analyzing the dataset to understand the distribution of features, identifying correlations, and visualizing data trends.
- **Feature Engineering** : Creating new features or modifying existing ones to improve the model's performance. This may include aggregating transaction history, generating statistical features, and encoding categorical variables.
- **Model Selection** : Evaluating different machine learning algorithms such as Random Forest, Logistic Regression, Support Vector Machine to determine the best model for fraud detection.
- **Model Training** : Training the selected model on a subset of the dataset while ensuring to handle imbalanced classes, as fraudulent transactions are typically rare compared to legitimate ones.
- **Model Evaluation** : Assessing the model's performance using metrics like accuracy, precision, recall, F1-score to ensure it effectively identifies fraudulent transactions.

Leveraging advanced machine learning techniques, this project aims to enhance the detection of fraudulent credit card transactions, thereby reducing financial losses and improving security for credit card users and financial institutions.
## About the Data
The dataset contains transactions made by credit cards in September 2013 by European cardholders.

The dataset presents transactions that occurred in two days, where it has 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, the positive class (frauds) account for 0.172% of all transactions.
The data contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues,the original features (names) and more background information about the data are not given. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Source: (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Authors

- Machine Learning Group - ULB (Owner)
- Andrea (Collaborator)
## Prerequisites

Before running this project, please ensure having the following software and libraries installed on machine:

- **Python 3**: Ensure having the latest version of Python 3 installed.

- **Anaconda**: Anaconda is a platform that includes many essential libraries and tools. Installing Anaconda will provide with:

    - **Jupyter Notebook**: For creating and sharing documents containing live code, equations, visualizations, and narrative text.
    - **Libraries**: The following libraries come pre-installed with   Anaconda and are required for this project:
        - **pandas**: For data manipulation and analysis.
        - **numpy**: For numerical computations.
        - **matplotlib**: For plotting and visualization.
        - **seaborn**: For statistical data visualization.
        - **scikit-learn (sklearn)**: For machine learning.
If Anaconda is not installed, you can download and install it from the official Anaconda website (https://www.anaconda.com/products/navigator). Follow the installation instructions for your operating system.


## Applied Machine Learning Algorithms

### Random Forest

Random Forest is an ensemble learning model primarily used for classification and regression tasks. It can effectively handle imbalanced classes by giving more weight to the minority class. The model builds multiple decision trees and aggregates their predictions, helping to mitigate the impact of noise and outliers, which are common in transaction data.

Random Forest can efficiently manage large datasets and high-dimensional data, making it suitable for the extensive and complex transaction data used in credit card fraud detection.

The model has been trained on balanced training data obtained using the oversampling technique (SMOTE), resulting in improved performance.



![Illustration-of-random-forest-trees](https://github.com/user-attachments/assets/68e4addc-c425-4978-9958-95e5b821247e)

#### Performance Metrics to Define Model Performance 


![random forest score](https://github.com/user-attachments/assets/f0ec649d-5df6-4537-855e-1660669ee007)

In the report, we observed a high recall score of 93%, indicating a low false negative rate. This means the model is highly effective at correctly predicting fraudulent transactions. Additionally, the model achieved a precision score of 93%, ensuring a low false positive rate, which enhances the model's overall accuracy.


#### ROC Curve and AUC Score of the model

![ROC AUC RANDOM Forest](https://github.com/user-attachments/assets/bdaec4b4-6a8d-4da0-a298-677e01823ed6)


The ROC curve above demonstrates the model's high accuracy, with an AUC score of approximately 0.99

### Logistic Regression

Logistic regression is effective for predicting fraudulent credit card transactions because it handles binary classification well, distinguishing between fraud and non-fraud. It provides probabilistic outputs, allowing easy interpretation of results. Logistic regression is also efficient with large datasets, and can handle multicollinearity effectively, making it suitable for financial data analysis.

The model has been trained on balanced training data obtained using the oversampling technique (SMOTE), resulting in improved performance.



![lr example](https://github.com/user-attachments/assets/55e1ce9d-25fa-408c-b7c7-74fe4e4bd90c)


#### Performance Metrics to Define Model Performance

![lr_score](https://github.com/user-attachments/assets/f8760398-7fae-4b00-ad4b-943ed09283a6)


In the picture above, although we have achieved the highest recall score, the precision score is relatively low, indicating that the model has not effectively reduced the false positive rate.

#### ROC Curve and AUC Score of the model

![lr_roc](https://github.com/user-attachments/assets/95c74a0c-e990-4a36-8f4f-eac93637e7cf)


The ROC curve above demonstrates the model's high accuracy, with an AUC score of approximately 0.99

### Support Vector Machine

Support Vector Machines (SVM) are effective for predicting credit card fraud due to their ability to handle high-dimensional data and distinguish between classes with clear margins. They are robust against overfitting, especially in high-dimensional spaces, which is crucial for detecting rare fraudulent transactions. Additionally, SVMs can efficiently process non-linear relationships in data using kernel functions, enhancing their predictive accuracy for fraud detection.

The model has been trained on balanced training data obtained using the oversampling technique (SMOTE), resulting in improved performance.



![An-example-of-an-SVM-classification-using-the-RBF-kernel-The-two-classes-are-separated](https://github.com/user-attachments/assets/f1d38f9c-5c71-42dd-880f-aa56f90a51a6)

#### Performance Metrics to Define Model Performance 


![svm_score](https://github.com/user-attachments/assets/46ae0b2f-aa6f-4f18-bdb4-be358013a411)

The model has demonstrated a low recall score and the lowest accuracy score, indicating its poor performance in predicting outcomes.

#### ROC Curve and AUC Score of the model


![ROC_SVM](https://github.com/user-attachments/assets/5b5f9ae4-ce05-45b0-add1-da97e8ead9dd)

The curve in the image is nearly aligned with the diagonal line, which represents the performance of a naive model. This results in a relatively poor ROC curve with an AUC score of 0.57.



## Conclusions

- The Random Forest model achieves a balanced performance with both precision and recall scores of 0.93. In contrast, Logistic Regression demonstrates a high recall close to 1 but has a significantly lower precision of 0.07, indicating that it tends to classify almost every future transaction as fraudulent, which limits its practical usefulness. Meanwhile, the SVM model shows a recall score of 0.50 and a precision score of 0, suggesting that it performs poorly overall and is not effective for this particular task.

- The Random Forest model achieved an impressive accuracy score of 99.98%, slightly outperforming the Logistic Regression model, which had an accuracy of 97.89%. In contrast, the Support Vector Machine (SVM) model demonstrated a significantly lower accuracy of 56.85%.

- Overall, the Random Forest method outperformed the other two models in detecting fraudulent transactions.

- We can also improve on this accuracy by increasing the sample size or using deep learning algorithms however at the cost of computational expense. We can also use complex anomaly detection models to get better accuracy in determining more fraudulent cases.