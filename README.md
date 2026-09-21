Makoding — DataLab Pro

An interactive data-science workspace for preparing data, exploring patterns, engineering features, building machine-learning models, and discovering structure in data.

Makoding is a modular Python data-science package powering DataLab Pro, a Streamlit-based analytical workspace.

DataLab Pro brings the main stages of a practical data-science workflow into one environment:

Load → Clean → Explore → Visualize → Engineer → Model → Discover → Export

The project is deliberately structured so that analytical logic lives in reusable Python modules rather than being embedded directly inside the Streamlit interface. This makes the system easier to test, maintain, extend, audit, and deploy.

What DataLab Pro Does

DataLab Pro provides an interactive workspace for structured datasets. Users can:

Upload CSV, TSV, and Excel datasets

Load supported datasets from remote HTTP/HTTPS URLs

Inspect dataset structure and data quality

Clean missing values and duplicate records

Trim text whitespace

Produce cleaning reports

Perform exploratory data analysis

Examine correlations, outliers, duplicates, unique values, data types, and memory usage

Create statistical and analytical visualizations

Explore numeric distributions and categorical frequencies

Explore relationships between variables

Generate model diagnostic plots

Engineer numerical, categorical, datetime, cyclical, frequency, polynomial, difference, product, and ratio features

Scale and transform numerical variables

Encode categorical variables

Detect correlated and low-variance features

Train supervised classification and regression models

Evaluate model performance

Inspect feature importance

Perform K-Means, Agglomerative Clustering, and DBSCAN

Apply Principal Component Analysis (PCA)

Inspect clustering diagnostics and summaries

Export cleaned datasets and analytical outputs

Generate self-contained HTML analytical reports

Export machine-readable dataset profiles as JSON

Application Workflow

The Streamlit interface is organized into six main workspaces:

1. Overview

A high-level view of the current dataset, including:

Dataset dimensions

Missing cells

Duplicate rows

Data types

Dataset preview

Visual exploration

Quick analytical summaries

Export options

2. EDA

The exploratory-analysis workspace provides deeper inspection of the dataset:

Numeric summaries

Categorical summaries

Missing-value analysis

Correlation analysis

Outlier analysis

Duplicate analysis

Unique-value analysis

Memory-usage analysis

Data-type summaries

Numeric distribution plots

Categorical frequency plots

Numeric relationship exploration

3. Feature Engineering

Users can interactively select variables and apply reusable transformations without leaving the application.

Supported operations include:

Numeric scaling

Categorical encoding

Missing-value indicators

Numeric binning

Target encoding

Log transformations

Square-root transformations

Power transformations

Datetime features

Cyclical features

Frequency features

Difference features

Polynomial features

Product features

Ratio features

Correlated-feature detection

Low-variance feature detection

The feature-engineering layer is designed to preserve input data and handle common pandas dtype issues, including scaling integer-valued columns into floating-point outputs.

4. Model Builder

The Model Builder supports both classification and regression.

Users can:

Select a target variable

Select classification or regression

Choose a model

Configure train/test splitting

Configure the random state

Train the model

Review evaluation metrics

Inspect feature importance

Review predictions

Categorical inputs can be prepared through one-hot encoding before model training.

Model diagnostics include appropriate plots for the selected task, including:

Confusion matrix

ROC curve

Precision-recall curve

Residual plots

Predicted-versus-actual plots

Feature-importance plots

5. Unsupervised Learning

The Unsupervised workspace allows users to explore structure without specifying a target variable.

Supported methods:

K-Means

Agglomerative Clustering

DBSCAN

Principal Component Analysis (PCA)

Supporting analysis includes:

Standardization

Silhouette scoring

K-Means inertia

Elbow analysis

Cluster summaries

K-distance analysis

PCA explained-variance analysis

6. Downloads

The application provides working exports for the current analytical workflow:

Cleaned dataset as CSV

Self-contained analytical report as HTML

Machine-readable dataset profile as JSON

The HTML report embeds generated charts so the report can be opened independently of the Streamlit application.

Architecture

DataLab Pro separates the user interface from reusable analytical functionality.

                         ┌──────────────────────────┐
                         │      Streamlit App       │
                         │         app.py            │
                         └────────────┬─────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
      ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
      │   Data I/O  │          │   Cleaning  │          │     EDA     │
      │ data_io.py  │          │ cleaning.py │          │   eda.py    │
      └─────────────┘          └─────────────┘          └─────────────┘
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
      ┌──────────────┐         ┌─────────────┐          ┌──────────────┐
      │   Feature    │         │   Modeling  │          │ Unsupervised │
      │ Engineering  │         │ modeling.py │          │unsupervised.py│
      │feature_      │         └─────────────┘          └──────────────┘
      │engineering.py│
      └──────────────┘
             │
             ▼
      ┌──────────────────────┐
      │ Visualization /      │
      │ Reporting / Exports  │
      └──────────────────────┘
             │
             ▼
      ┌──────────────────────┐
      │ Results & Insights   │
      └──────────────────────┘

The Streamlit application is primarily responsible for:

User interaction

Navigation

Configuration controls

Application state

Displaying analytical results

Connecting reusable package functions into an end-to-end workflow

The makoding package contains the reusable analytical functionality.

Project Structure

Makoding/
│
├── app.py
├── README.md
├── assets/
├── .vscode/
├── tests/
│   ├── __init__.py
│   ├── test_cleaning.py
│   ├── test_data_io.py
│   ├── test_eda.py
│   ├── test_feature_engineering.py
│   ├── test_modeling.py
│   ├── test_unsupervised.py
│   └── test_visualization.py
│
└── makoding/
    ├── __init__.py
    ├── config.py
    ├── cleaning.py
    ├── data_io.py
    ├── eda.py
    ├── feature_engineering.py
    ├── logging_config.py
    ├── modeling.py
    ├── styling.py
    ├── unsupervised.py
    └── visualization.py

Core Modules

data_io.py

Handles dataset ingestion and validation.

Supported sources include:

CSV

TSV

Excel .xlsx

Excel .xls

Remote HTTP/HTTPS dataset URLs

Example:

from makoding import data_io

df = data_io.load_dataframe(file, filename)

Remote dataset example:

from makoding import data_io

df = data_io.load_csv_url("https://example.com/data.csv")

The module validates supported extensions, upload size, URL schemes, and dataset contents before returning a DataFrame.

cleaning.py

Provides reusable dataset-cleaning functionality.

The cleaning workflow supports:

Missing-value handling

Duplicate removal

Text whitespace trimming

Data-quality reporting

Example:

from makoding import cleaning

cleaned_df, report = cleaning.clean_frame(
    df,
    missing="Keep",
    remove_duplicates=True,
    trim_whitespace=True,
)

The cleaning process returns both the cleaned DataFrame and a structured cleaning report.

eda.py

Provides framework-independent exploratory data-analysis utilities.

Available analysis includes:

Dataset overview

Missing-value summary

Numeric summary

Categorical summary

Correlation matrix

Data-type summary

Duplicate-row analysis

Unique-value analysis

Outlier analysis

Memory-usage analysis

Complete EDA reports

Example:

from makoding import eda

overview = eda.dataset_overview(df)
numeric = eda.numeric_summary(df)
missing = eda.missing_values_summary(df)
correlations = eda.correlation_matrix(df, method="pearson")

feature_engineering.py

Provides reusable preprocessing and feature-generation utilities.

Available functionality includes:

Numeric scaling

Categorical encoding

Missing indicators

Numeric binning

Target encoding

Log, square-root, and power transformations

Datetime features

Cyclical features

Frequency features

Difference features

Polynomial features

Product features

Ratio features

Correlated-feature detection

Low-variance feature detection

Example:

from makoding import feature_engineering

scaled = feature_engineering.scale_numeric_features(
    df,
    columns=["sales", "price"],
    method="standard",
)

Categorical encoding example:

encoded = feature_engineering.encode_categorical_features(
    df,
    columns=["category", "region"],
    method="onehot",
)

Scaling returns floating-point values for transformed numeric columns, including when the source columns are integer-valued. The original input DataFrame is not mutated.

modeling.py

Provides supervised machine-learning functionality for classification and regression.

Classification models include:

Logistic Regression

Random Forest

Gradient Boosting

XGBoost

LightGBM

CatBoost

Regression models include:

Linear Regression

Random Forest

Gradient Boosting

XGBoost

LightGBM

CatBoost

The modelling module provides:

Train/test splitting

Model fitting

Prediction

Classification evaluation

Regression evaluation

Cross-validation

Feature importance

Grid search

Random search

Example:

from makoding import modeling

X_train, X_test, y_train, y_test = modeling.train_test_split_data(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=True,
)

result = modeling.fit_model(
    X_train,
    y_train,
    task="classification",
    model_name="random_forest",
    random_state=42,
)

predictions = modeling.predict(result.model, X_test)
metrics = modeling.evaluate_classification(y_test, predictions)

unsupervised.py

Provides reusable unsupervised-learning functionality.

Supported methods include:

K-Means

Agglomerative Clustering

DBSCAN

PCA

Additional utilities include:

Standardization

Silhouette scoring

K-Means inertia

Elbow analysis

Cluster summaries

K-distance analysis

The module uses structured result objects so clustering and dimensional-reduction outputs can be consumed consistently by the application and tests.

Example:

from makoding import unsupervised

scaled = unsupervised.standardize_features(df)

result = unsupervised.fit_kmeans(
    scaled,
    n_clusters=3,
    random_state=42,
)

visualization.py

Provides Matplotlib-based visualization utilities. The functions return matplotlib.figure.Figure objects rather than displaying plots directly, keeping the visualization layer reusable outside Streamlit.

Visualization families include:

Numeric distributions

Categorical counts

Scatter plots

Box plots by group

Category heatmaps

Confusion matrices

ROC curves

Precision-recall curves

Residual plots

Predicted-versus-actual plots

Feature-importance plots

This separation allows the same chart functions to be used by the Streamlit interface, tests, and report-generation workflow.

styling.py

Contains the DataLab Pro visual design system and reusable Streamlit presentation components.

The visual identity retains a restrained Lake Victoria-inspired direction while prioritizing readability, analytical clarity, and a professional application interface.

config.py

Centralizes application metadata, upload limits, supported file extensions, URL configuration, and model defaults.

logging_config.py

Provides the application's logging configuration so data-loading and analytical operations can be monitored without embedding logging setup throughout the application.

Installation

Requirements

The project currently targets Python 3.12.

Create the Conda environment:

conda create -n Makoding python=3.12 -y

Activate it:

conda activate Makoding

Navigate to the project:

cd C:\Users\admin\Makoding\Development

Install dependencies:

pip install -r requirements.txt

The environment uses packages including:

Streamlit

Pandas

NumPy

Scikit-learn

SciPy

Matplotlib

Plotly

OpenPyXL

Requests

XGBoost

LightGBM

CatBoost

Pytest

Running DataLab Pro

From the project root:

streamlit run .\app.py

Then open the local Streamlit address shown in the terminal, normally:

http://localhost:8501

Running the Tests

Makoding follows a test-driven development approach. The reusable analytical modules are tested independently of the Streamlit interface.

Run the complete suite:

python -m pytest -q

The test suite covers:

Data loading and validation

Data cleaning

Exploratory analysis

Feature engineering

Supervised modelling

Unsupervised learning

Visualization

Regression cases for important pandas/scikit-learn edge cases

Run individual test modules when debugging:

python -m pytest -q .\tests\test_cleaning.py
python -m pytest -q .\tests\test_data_io.py
python -m pytest -q .\tests\test_eda.py
python -m pytest -q .\tests\test_feature_engineering.py
python -m pytest -q .\tests\test_modeling.py
python -m pytest -q .\tests\test_unsupervised.py
python -m pytest -q .\tests\test_visualization.py

Verification

The package modules can be checked independently of Streamlit:

python -c "from makoding import cleaning, data_io, eda, feature_engineering, modeling, styling, unsupervised, visualization; print('All Makoding modules imported successfully')"

Expected output:

All Makoding modules imported successfully

Syntax-check the application:

python -m py_compile .\app.py

Compile the package modules as well:

python -m py_compile .\makoding\*.py

Development Philosophy

Makoding is being developed around several principles.

Modular

Analytical functionality belongs inside reusable package modules rather than being embedded directly in the Streamlit interface.

Testable

Core analytical functions are tested independently from the user interface.

Auditable

Data transformations and model outputs should be understandable and traceable.

Extensible

The package structure makes it possible to add new analytical capabilities without turning app.py into a monolithic application.

Practical

The objective is not simply to demonstrate individual machine-learning algorithms. DataLab Pro is intended to support an end-to-end analytical workflow from ingestion through interpretation and export.

Validation and Error Handling

The application validates data before analytical operations where appropriate.

Examples include validation of:

Unsupported file formats

Invalid URLs

Oversized uploads

Empty datasets

Missing target variables

Missing model features

Invalid regression targets

Insufficient observations

Invalid clustering configurations

Unsupported transformation parameters

Invalid model configuration

Optional machine-learning libraries are loaded only when required by the selected model. This keeps the core package modular while supporting a broader model catalogue.

Technology Stack

Technology

Purpose

Python

Core programming language

Pandas

Data manipulation

NumPy

Numerical computing

Scikit-learn

Machine learning and preprocessing

SciPy

Scientific computing

Matplotlib

Statistical and model visualizations

Plotly

Interactive visualization support

Streamlit

Interactive application interface

XGBoost

Gradient-boosted machine learning

LightGBM

Gradient-boosted machine learning

CatBoost

Gradient-boosted machine learning

Requests

Remote dataset loading

OpenPyXL

Excel .xlsx support

Pytest

Automated testing

Git / GitHub

Version control and collaboration

Current Versions

Makoding: 0.1.0

DataLab Pro: 1.0.0

Roadmap

Potential future development areas include:

More advanced visualization

Automated data-quality profiling

Automated preprocessing pipelines

Model comparison interfaces

Expanded hyperparameter optimization

Model persistence and loading

Explainable AI

Advanced feature-selection workflows

Time-series modelling

Additional clustering algorithms

Automated reporting

Persistent projects and datasets

User authentication

Production deployment

API access

MLOps capabilities

The current modular architecture is intended to allow these capabilities to be introduced incrementally without replacing the existing analytical modules.

Project Status

Makoding / DataLab Pro is an actively developed project with a working end-to-end analytical workflow:

Dataset ingestion
      ↓
Data cleaning
      ↓
Exploratory analysis
      ↓
Visualization
      ↓
Feature engineering
      ↓
Supervised modelling
      ↓
Unsupervised learning
      ↓
Diagnostics and interpretation
      ↓
Export and reporting

The core analytical modules are independently testable, while the Streamlit application provides a unified interface for interacting with them.

Repository

Source code:

https://github.com/Odingcom/Makoding

License

Makoding / DataLab Pro is released under the MIT License.

Copyright (c) 2026 George Omondi Oding

The MIT License permits others to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the license conditions. The software is provided without warranty.

The complete license text is available in the repository root as LICENSE.

Note: Third-party dependencies used by the project remain subject to their respective licenses.

For an open-source project intended for public peer review, GitHub recommends including a dedicated LICENSE file so the permissions granted to other users are explicit.

Makoding

DataLab Pro

Collect. Clean. Explore. Visualize. Engineer. Model. Discover. Export.

Inspired by the waters of Lake Victoria.