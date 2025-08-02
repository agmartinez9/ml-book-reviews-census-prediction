# ACDD – Applied Classification with Diverse Datasets

This project applies machine learning techniques to analyze two distinct datasets: book reviews and U.S. census data. The goal is to explore patterns, build predictive models, and evaluate their performance using classification algorithms.

## Project Structure

- `DefineAndSolveMLProblem.ipynb` — Main Jupyter notebook with data analysis, model training, and evaluation
- `bookReviewsData.csv` — Dataset containing book review information
- `censusData.csv` — Dataset with demographic information
- `airbnbListingsData.csv` — Dataset containing Airbnb property listings with pricing, location, and host information

## Objectives

- Perform data cleaning and preprocessing
- Apply supervised learning algorithms (e.g., logistic regression, decision trees)
- Evaluate models using appropriate performance metrics
- Draw insights from model predictions

## Results Summary
  Successfully trained machine learning models using the census income dataset.
  Explored classification algorithms such as Logistic Regression and Random Forest Classifier.
  Achieved 86.3% accuracy on the test set for predicting income bracket (≤$50K vs >$50K).
  Key metrics:
  Accuracy: 86.3%
  Precision: 69.6%
  Recall: 64.9%
  F1-score: 67.2%
  Data preprocessing steps such as handling missing values and one-hot encoding improved model performance.
  The Random Forest model provided interpretable insights into feature importance with relationship status and capital gains as top predictors.
  Visualizations include:
  Confusion matrix showing model prediction accuracy
  Feature importance plot highlighting top contributing variables
  Income distribution charts showing class imbalance
  Correlation heatmap of numerical features
