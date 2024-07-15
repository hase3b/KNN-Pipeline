# KNN-Pipeline
This repository contains an implementation of a comprehensive machine-learning pipeline tailored for both classification and regression tasks. The pipeline encompasses various stages, including data collection, cleaning, transformation, exploratory data analysis (EDA), feature selection, dimensionality reduction, data splitting, cross-validation, model selection, training, and evaluation. Currently, the pipeline only supports the K-Nearest Neighbors (KNN) algorithm since the aim of the project is to demonstrate the pipeline execution for such tasks and the pipeline can be easily modified to support other algorithms.

### Repository Structure
This repository contains a Jupyter Notebook for the pipeline's support and master functions along with the pipeline's execution on multiple datasets including both classification and regression problems. Further, an excel file shows the compiled results.

## Pipeline Flow
1) Fetch Data: Data is sourced from the UCI Machine Learning Repository.
2) Data Cleaning: Treatment of missing and duplicate values to ensure data quality.
3) Exploratory Data Analysis (EDA): Numerical Summary: Summarizing numerical features to understand data distribution and characteristics.
4) EDA: Class Balance (%): Examining class distribution to identify potential imbalances (for classification tasks only).
5) EDA: Histogram & Box Plot of Important Numerical Features: Visualizing key numerical features to detect patterns and anomalies.
6) Outliers Detection & Treatment: Not Implemented Yet: Placeholder for future development.
7) Data Transformation: Encoding categorical variables and standardizing numerical features (both standard and min-max scaler available).
8) Feature Selection: Selecting the most relevant features using various methods.
   * Filter Methods: Includes correlation heatmap and mutual information.
   * Wrapper Methods:
       * Forward Selection: Iteratively adding features that improve model performance.
       * Backward Selection: Iteratively removing features that degrade model performance.
    * Dimensionality Reduction (PCA): Reducing the feature space while retaining most of the variance.
9) PCA Visualization: Visualizing the principal components to understand data separability (for classification tasks only).
10) Data Splitting & Cross-Validation: Splitting the data into training and testing sets and implementing cross-validation to ensure robust model evaluation.
11) Model Selection (Lazy Learner): Utilizing a lazy learner approach to evaluate multiple models quickly and select the best performing one.
12) Model Training (KNN): Training the K-Nearest Neighbors (KNN) model.
    * Non-CV:
       * Classification: Training without cross-validation.
       * Regression: Training without cross-validation.
    * CV:
       * Classification: Training with cross-validation.
       * Regression: Training with cross-validation.
13) Model Evaluation: Evaluating model performance using various metrics.
    * Non-CV:
       * Classification: Precision, recall, F1-score.
       * Regression: Mean squared error (MSE), mean absolute error (MAE), root mean squared error (RMSE), R-squared, Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), p-value.
    * CV:
       * Classification: Evaluating classification performance with cross-validation.
       * Regression: Evaluating regression performance with cross-validation.

### Future Work
* Implement outliers detection and treatment.
* Expand the pipeline to include additional models and hyperparameter tuning.
* Enhance the visualization capabilities for better data insights.
