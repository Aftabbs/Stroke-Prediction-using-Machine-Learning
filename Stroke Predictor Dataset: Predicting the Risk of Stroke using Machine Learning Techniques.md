# Stroke-Prediction-using-Machine-Learning
Stroke Predictor Dataset
This project aims to predict whether an individual is at risk of having a stroke based on various demographic, lifestyle, and health-related factors. The dataset used for this project is the Stroke Prediction Dataset from Kaggle.

# Table of Contents
Installation
Usage
Data Cleaning and Preprocessing
Exploratory Data Analysis
Model Training and Evaluation
Conclusion

# Installation
To run this project, you need to have Jupyter Notebook installed on your computer along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
You can install these libraries by running the following command:
pip install pandas numpy scikit-learn matplotlib seaborn
Alternatively, you can use the Anaconda distribution of Python (Jupyter Notebook), which comes with all the necessary libraries pre-installed.

open the file 
jupyter notebook stroke_prediction.ipynb

# Data Cleaning and Preprocessing
The dataset contains missing values, categorical variables, and outliers, which need to be addressed before building a machine learning model. In the notebook, we perform the following data cleaning and preprocessing steps:

- Drop the ID column, which is not relevant for the analysis.
- Handle missing values in the BMI column using the mean value.
- Convert categorical variables to numerical using one-hot encoding.
- Detect and remove outliers using the IQR method.
- Exploratory Data Analysis
In this section, we explore the relationship between the target variable (stroke) and other features in the dataset. We use visualizations such as histograms, bar plots, and scatter plots to gain insights into the data. Some of the questions we answer in this section are:

![image](https://user-images.githubusercontent.com/112916888/224869440-11794154-33ba-4fcf-b792-382c9d5bd995.png)


![image](https://user-images.githubusercontent.com/112916888/224869555-88da2e80-9229-409c-8956-e0ebdd15df5a.png)

![image](https://user-images.githubusercontent.com/112916888/224869691-2977a9c3-4fe5-428e-ab15-4ccc37a0e062.png)

![image](https://user-images.githubusercontent.com/112916888/224869762-db04c75d-a1c6-42df-bc27-3b91a843552f.png)


What is the distribution of the target variable?
![image](https://user-images.githubusercontent.com/112916888/224869852-ed560840-a2fd-4037-9ecc-33675ea2b897.png)


How are the features related to the target variable?
Are there any correlations between the features?
Model Training and Evaluation
In this section, we build several machine learning models to predict the probability of stroke in an individual. We use the following models:

- Decision Tree
- Random Forest
- XGBoost

Decision Tree and Random Forest model were not upto the Expectations hence XGB performed well before and After Tuning
We also perform hyperparameter tuning using GridSearchCV to find the best parameters for each model. The models are evaluated using various metrics such as accuracy, precision, recall, F1-score.

# Conclusion
Based on our analysis, we find that the XGB model performs the best with an accuracy of 0.91 and an f1score of 0.90. The top factors that contribute to the risk of stroke are age, hypertension, and heart disease. This project shows how machine learning can be used to predict the likelihood of stroke in individuals and can help in early detection and prevention of the disease.





