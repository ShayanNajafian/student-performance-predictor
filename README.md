# 🎓 Student Performance Predictor

A machine learning project that predicts a student's final Mathematics grade using demographic, academic, family, social, and school-related information.

The project covers the complete data-science workflow:

- Exploratory data analysis
- Data preprocessing
- Model comparison and cross-validation
- Hyperparameter tuning
- Model evaluation
- Model persistence
- Interactive prediction with Streamlit

## Project Objective

The target variable is **`G3`**, the student's final Mathematics grade on a **0–20 scale**.

Two prediction scenarios were evaluated:

### Full Model

Uses all available predictors, including:

- `G1` — first-period grade
- `G2` — second-period grade

This scenario represents prediction when earlier academic performance is already known.

### Early-Prediction Model

Excludes `G1` and `G2`.

This scenario evaluates how well final performance can be estimated using demographic, behavioral, family, and school-related variables alone.

## Dataset

The project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

The data contains information about students from two Portuguese secondary schools and includes variables related to:

- Demographics
- Family background
- Study habits
- School support
- Social behavior
- Absences
- Previous academic performance
- Final grades

Dataset source:

https://archive.ics.uci.edu/dataset/320/student+performance

The Mathematics dataset used in this project is:

```text
student-mat.csv
```

The raw dataset is not stored directly in the repository.

Place the downloaded file at:

```text
data/raw/student-mat.csv
```

## Project Structure

```text
student-performance-predictor/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── raw/
│       └── student-mat.csv
│
├── models/
│   ├── model_metadata.json
│   └── student_performance_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_selection_evaluation.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Exploratory Data Analysis

The EDA investigates:

- Dataset structure and data quality
- Numerical and categorical variables
- Distribution of the final grade
- Relationships between student characteristics and `G3`
- Correlations between `G1`, `G2`, and `G3`
- Study time, failures, absences, and other behavioral variables

A major finding was the strong relationship between previous-period grades and the final grade.

## Preprocessing

Numerical and categorical variables are processed separately using a Scikit-learn `ColumnTransformer`.

### Numerical Features

- Median imputation
- Standard scaling

### Categorical Features

- Most-frequent-value imputation
- One-hot encoding
- Unknown category handling

Preprocessing is fitted only on training data to reduce the risk of data leakage.

## Model Selection

The following regression models were compared:

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Models were evaluated using **5-fold cross-validation**.

The primary model-selection metric was **Mean Absolute Error (MAE)**, supported by:

- Root Mean Squared Error (RMSE)
- R²
- Train-validation error gap

Random Forest produced the strongest performance in both feature scenarios.

## Final Results

| Scenario | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Early Prediction | 3.036 | 3.813 | 0.291 |
| Full Model | **1.209** | **2.006** | **0.804** |

The Full model reduced MAE substantially compared with the Early model.

This indicates that **previous academic performance provides a large amount of predictive information about final student performance**.

## Final Model

The deployed model is a tuned **Random Forest Regressor** using the Full feature set.

Selected hyperparameters:

```text
n_estimators      = 500
max_depth         = None
max_features      = 1.0
min_samples_leaf  = 2
```

The complete preprocessing and prediction pipeline is stored in:

```text
models/student_performance_model.pkl
```

This allows raw user input to pass through the same preprocessing steps used during model training before generating a prediction.

## Feature Importance

For the Early-Prediction model, previous failures and school absences emerged as two of the strongest predictive features.

Feature importance should be interpreted as a measure of how much the fitted model relies on a variable for prediction, rather than evidence of a causal relationship.

## Streamlit Application

The project includes an interactive Streamlit application that allows users to enter student information and receive a predicted final Mathematics grade.

The application:

- Collects academic, personal, school, family, and lifestyle information
- Converts user inputs into the expected model format
- Loads the saved Scikit-learn pipeline
- Generates a prediction for `G3`
- Displays the model's test MAE and R² for context

## Run Locally

Clone the repository and move into the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows using Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit in your browser.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter
- Git
- GitHub

## Limitations

The dataset is relatively small and represents students from only two Portuguese secondary schools, so model performance should not be assumed to generalize to all students or education systems.

The model is intended as a machine-learning demonstration rather than a definitive assessment of an individual student's ability or academic potential.

Model predictions represent statistical relationships in the training data and should not be interpreted as causal conclusions.

## Live Demo

Try the deployed application:

https://student-performance-predictor-nloywuaavlu4dzr2gvkob2.streamlit.app