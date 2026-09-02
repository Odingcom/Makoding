# Makoding

**Makoding** is a Python-based data science application designed to simplify dataset ingestion, validation, cleaning, exploration, feature engineering, and machine-learning workflows.

The project is being developed as a reusable data-science toolkit with a Streamlit interface.

## Features

### Data ingestion

Makoding currently supports:

* CSV files
* TSV files
* Excel `.xlsx` files
* Excel `.xls` files
* Loading datasets from uploaded files
* Loading datasets from HTTP/HTTPS URLs
* File-extension validation
* Upload-size validation
* Dataset row-limit validation
* URL validation
* Encoding fallback for CSV files
* Protection against oversized URL responses

### Data cleaning

The cleaning module currently supports:

* Trimming whitespace from column names
* Trimming whitespace from string values
* Removing duplicate rows
* Keeping missing values
* Dropping rows containing missing values
* Filling numeric missing values with the median
* Filling numeric missing values with the mean
* Filling missing values using the column mode
* Detailed cleaning reports
* Protection against modification of the original DataFrame

### Testing

The project currently contains automated tests covering the data-cleaning and data-ingestion modules.

The current test suite contains **89 tests**, all of which pass.

## Project structure

```text
Makoding/
│
├── app.py
├── README.md
├── .gitignore
│
├── makoding/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── config.py
│   ├── data_io.py
│   ├── eda.py
│   ├── features.py
│   ├── logging_config.py
│   ├── modeling.py
│   └── styling.py
│
└── tests/
    ├── __init__.py
    ├── test_cleaning.py
    └── test_data_io.py
```

## Requirements

Makoding is developed with Python 3.12.

The main environment currently uses packages including:

* pandas
* NumPy
* Streamlit
* openpyxl
* SciPy
* Plotly
* CatBoost
* LightGBM
* XGBoost
* pytest

A formal dependency file will be added as the project develops.

## Installation

Clone the repository:

```bash
git clone https://github.com/Odingcom/Makoding.git
cd Makoding
```

Create and activate the Conda environment:

```bash
conda create -n Makoding python=3.12 -y
conda activate Makoding
```

Install the required packages:

```bash
pip install pandas numpy streamlit openpyxl scipy plotly catboost lightgbm xgboost pytest
```

## Running the application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will provide the local Streamlit address in the terminal.

## Running the tests

Run the complete test suite:

```bash
python -m pytest -v
```

Run only the cleaning tests:

```bash
python -m pytest tests/test_cleaning.py -v
```

Run only the data-ingestion tests:

```bash
python -m pytest tests/test_data_io.py -v
```

## Development status

Makoding is under active development.

### Completed

* Project structure established
* Configuration module established
* Logging configuration established
* Data ingestion and validation implemented
* Data cleaning functionality implemented
* Automated tests implemented
* 89 tests currently passing
* Git/GitHub repository established

### In development

Future development will expand:

* Exploratory data analysis
* Feature engineering
* Machine-learning workflows
* Model evaluation
* Visualization
* Streamlit user interface
* Configuration and customization
* Documentation
* Automated CI/CD testing
* Deployment

## Quality assurance

The project uses `pytest` for automated testing.

Before pushing changes, run:

```bash
python -m pytest -v
```

All tests should pass before changes are committed.

## Repository

GitHub:

https://github.com/Odingcom/Makoding

## License

License information will be added as the project matures.
