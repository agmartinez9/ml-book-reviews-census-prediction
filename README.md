# ACDD – Applied Classification with Diverse Datasets
This project applies machine learning techniques to analyze two distinct datasets: book reviews and U.S. census data. The goal is to explore patterns, build predictive models, and evaluate their performance using classification algorithms.

## Project Structure
- `DefineAndSolveMLProblem.ipynb` — Main Jupyter notebook with data analysis, model training, and evaluation
- `bookReviewsData.csv` — Dataset containing book review information
- `censusData.csv` — Dataset with demographic information
- `airbnbListingsData.csv` — Dataset containing Airbnb property listings with pricing, location, and host information

## Methodology
The project follows a systematic machine learning lifecycle approach:
Data Exploration: Analyzed dataset structure, missing values, and class distribution
Data Preprocessing: Applied mode imputation for missing values, one-hot encoding for categorical variables, and feature scaling
Model Selection: Compared Logistic Regression (baseline) and Random Forest Classifier
Hyperparameter Tuning: Used Grid Search CV with 3-fold cross-validation to optimize Random Forest parameters
Evaluation: Assessed models using accuracy, precision, recall, and F1-score metrics with train/test split (80/20)

## Requirements
pandas==1.3.0
numpy==1.21.0
scikit-learn==1.0.0
matplotlib==3.4.0
seaborn==0.11.0
jupyter==1.0.0

## Key Findings
Random Forest outperformed Logistic Regression with 86.3% accuracy
Top predictive features: relationship status, capital gains, age, education level, and hours per week
Model effectively identifies high-income individuals with 69.6% precision
Hyperparameter tuning improved performance by 0.6%

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

## Installation Steps
    Clone the repository:
    bashgit clone https://github.com/agmartinez9/ml-book-reviews-census-prediction.git
    cd ml-book-reviews-census-prediction
    Install required packages:
    bashpip install -r requirements.txt
    Launch Jupyter Notebook:
    bashjupyter notebook DefineAndSolveMLProblem.ipynb


## Usage
Data Loading: Run the first cells to load and explore the census dataset
Preprocessing: Execute data cleaning and feature engineering sections
Model Training: Train both Logistic Regression and Random Forest models
Evaluation: Analyze model performance using confusion matrices and classification reports
Visualization: Generate feature importance plots and performance charts

## Next Steps
Implement advanced ensemble methods (XGBoost, LightGBM)
Explore feature engineering techniques (polynomial features, interaction terms)
Deploy model as REST API for real-time predictions
Conduct bias analysis to ensure fair AI practices
Expand to multi-class income classification
