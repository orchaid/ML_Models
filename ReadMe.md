# Machine Learning Projects

This repository showcases three comprehensive machine learning projects, each focusing on a distinct area: classification, regression, and unsupervised learning. These projects were undertaken to deepen understanding and practical skills in data preprocessing, model development, evaluation, and interpretation using real-world datasets.



## 1. Magic Gamma Telescope Classification

**Objective:**
Classify particles detected by the MAGIC (Major Atmospheric Gamma Imaging Cherenkov) telescope as either gamma rays or hadrons based on observed features.

**Dataset:**
[MAGIC Gamma Telescope Dataset](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope) from the UCI Machine Learning Repository.

**Techniques and Tools:**

* **Data Preprocessing:** scaled the data, normalized features and oversampled unbalanced data.
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and correlations.
* **Modeling:** Implemented Logistic Regression, and KNN, Naive Bayes, SVM and Neural Networks for classification.
* **Evaluation:** Assessed models using accuracy, precision, recall, F1-score, and ROC-AUC.
* **Hyperparameter Tuning:** Utilized GridSearchCV for optimal model parameters.

**Key Insights:**

* SVM outperformed other models with the highest accuracy and AUC.
* Feature importance analysis highlighted key variables influencing classification.

**Notebook:**
[View Notebook](https://github.com/orchaid/ML_Models/blob/main/Magic_Gamma_Telescope_Classification.ipynb)



## 2. Seoul Bike Sharing Demand Regression

**Objective:**
Predict the hourly demand for bike rentals in Seoul based on temporal and environmental features.

**Dataset:**
[Seoul Bike Sharing Demand Dataset](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) from the UCI Machine Learning Repository.

**Techniques and Tools:**

* **EDA:** Analyzed trends and seasonality in bike rentals.
* **Feature Engineering:** using visuals identified the features that has the biggest relationship with the target then dropped hour, day, month, and holiday indicators.
* **Modeling:** Applied Linear Regression, Random Forest Regressor, and neural networks Regressor.
* **Evaluation:** Measured performance using RMSE and R² metrics.
* **Visualization:** Plotted actual vs. predicted values to assess model fit.

**Key Insights:**

* Neural Net Regressor provided the most accurate predictions with the lowest RMSE (58914.29135837694).
* Temperature and hour of the day were significant predictors of bike demand.

**Notebook:**
[View Notebook](https://github.com/orchaid/ML_Models/blob/main/Seoul_Bike_Sharing_Demand_Regression.ipynb)



## 3. Seeds Dataset - Unsupervised Learning

**Objective:**
Cluster wheat seed samples and reduce dimensionality to uncover inherent patterns without labeled data.

**Dataset:**
[Seeds Dataset](https://archive.ics.uci.edu/ml/datasets/seeds) from the UCI Machine Learning Repository.

**Techniques and Tools:**

* **Data Preprocessing:** Standardized features for uniformity.
* **Dimensionality Reduction:** Applied Principal Component Analysis (PCA) to reduce features to two principal components.
* **Clustering:** Utilized KMeans clustering to group similar samples.
* **Visualization:** Plotted clusters in the PCA-reduced space for interpretability.

**Key Insights:**

* Optimal clustering achieved with three clusters, aligning with the known seed types.
* PCA effectively captured the variance, facilitating clear cluster separation.

**Notebook:**
[View Notebook](https://github.com/orchaid/ML_Models/blob/main/Seeds_Unsupervised_Learning.ipynb)



## Summary

These projects collectively enhanced proficiency in:

* **Data Preprocessing:** Handling missing values, feature scaling, and handling unbalanced classifiers.
* **Model Development:** Implementing and tuning various machine learning algorithms.
* **Evaluation Metrics:** Selecting appropriate metrics for classification and regression tasks.
* **Unsupervised Learning:** Applying clustering and dimensionality reduction techniques.
* **Visualization:** Creating insightful plots to interpret model results and data patterns.

Each notebook is self-contained and provides a step-by-step walkthrough of the machine learning pipeline, making them valuable resources for learning and reference.
