# Churn_Customer_Lifetime_Value_prediction

Quick summary of the readme file. First, there are some guidelines about project installation. Then, there are some notions about the dataset and decisions I made in the data preparation stage and model selection. At the end, I discuss my result.

---

## Requirements

- Python 3.x
- Libraries listed in `requirements.txt`

---

## Setup

### Clone the Repository
```bash
git clone https://github.com/Pmilivojevic/facial_expression_detection.git
cd facial_expression_detection
```

### Create a Virtual Environment
```bash
virtualenv env
source env/bin/activate
```

---

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
1. Execute the main script:

```bash
python main.py
```

2. Upon running, the project generates a structured output in the artifact folder, containing results from the pipelines.

---
## Notions about the dataset

I performed Exploratory Data Analysis in Jupyter Notebook EDA.ipynb and checked for missing values, numerical and categorical features, their distributions, and correlation.

features(columns): ftd_date, qp_date, total_handle, have missing values. Of which, total_handle is numerical and ftd_date, qp_date are datetime features.

Numerical features are: account_id(int64), tracker_id(int64), total_deposit(float64), total_handle(float64), total_ngr(float64).

Categorical features: brand_id(object), ben_login_id(object), player_reg_product(object).

Datetime features: activity_month(datetime64[ns]), reg_date(datetime64[ns]), ftd_date(datetime64[ns]), qp_date(datetime64[ns]).

One row of the datase represent player’s activity aggregated by month.

Net Gaming Revenue (NGR):
Proxy for player value; higher NGR = more profitable player for the sportsbook

months_active calculation - Difference in months between activity_month and ftd_date

Churn definition - Player churns if no activity for ≥ 2 months after latest activity

Early churn - Player leaves within N months after FTD (e.g. 2 months)

has_qp indicator - 1 if player placed a bet (non-null qp_date), else 0

days_ftd_to_qp - Negative if no QP — useful to spot quick vs. delayed engagement

tracker_id - Initially dropped due to high cardinality and sparse distribution

#### Assumptions — Model Selection
model 1: Logistic Regression (for early churn prediction) - Interpretable baseline model, easy to regularize, works well when features are standardized.

model 2: XGBoostClassifier (for churn probability) - Tree-based, handles nonlinear interactions, doesn’t need scaling, robust to multicollinearity.

#### Assumptions — Data Splitting
Split by temporal order ensures no leakage from future records. If not possible, split carefully by player to avoid splitting a player's history between train/test

Train-test split ratio: 80:20 (modifiable via config)

---
## Notions about my results

I've spent a lot of time trying to understand the data and what churn models should predict and how accordingly I should perform data engineering. My results are not good, maybe I didn't make the correct assumption, or I made some error in developing. I don't know. But with more time, I believe I would make it to work well.

