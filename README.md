# 📡 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?style=flat&logo=powerbi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-green?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

> End-to-end telecom churn prediction project using Python machine learning models and an interactive 3-page Power BI dashboard to identify high-risk customers and drive retention strategies.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tools & Technologies](#-tools--technologies)
- [Analysis Steps](#-analysis-steps)
- [Key Findings](#-key-findings)
- [ML Model Results](#-ml-model-results)
- [Business Recommendations](#-business-recommendations)
- [Dashboard Preview](#-dashboard-preview)
- [How to Run](#️-how-to-run)

---

## 📌 Project Overview

Customer churn costs telecom companies millions every year. Acquiring a new customer costs 5–7× more than retaining an existing one. This project:

- Analyses 7,032 telecom customer records to find churn patterns
- Builds ML models to predict which customers will leave
- Identifies the top factors driving churn
- Flags individual high-risk customers for retention teams
- Delivers a 3-page interactive Power BI dashboard

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | Telco Customer Churn |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,032 customers |
| **Columns** | 33 features |
| **Target** | `Churn Label` (Yes / No) |
| **Churn Rate** | 26.6% |

**Key Features:**
- `Contract` — Month-to-month, One year, Two year
- `Tenure Months` — How long the customer has been with the company
- `Monthly Charges` — Monthly bill amount
- `Internet Service` — Fiber optic, DSL, No
- `Tech Support` — Yes / No
- `Online Security` — Yes / No
- `Payment Method` — Electronic check, Mailed check, etc.

---

## 📁 Project Structure
```
telecom-churn-prediction/
│
├── data/
│   └── Telco_customer_churn.xlsx
│
├── output/
│   ├── churn_cleaned.csv
│   ├── churn_predictions.csv
│   ├── 01_churn_distribution.png
│   ├── 02_contract_churn.png
│   ├── 03_charges_churn.png
│   ├── 04_tenure_churn.png
│   ├── 05_internet_churn.png
│   ├── 06_model_comparison.png
│   ├── 07_roc_curve.png
│   ├── 08_confusion_matrix.png
│   └── 09_feature_importance.png
│
├── Screenshots/
│   ├── Churn Overview.png
│   ├── High Risk Customer.png
│   └── Retention Strategy.png
│
├── PowerBI/
│   └── Dashboard.pbix
│
├── churn_analysis.py
└── README.md
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Data cleaning, EDA, ML models |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualisations |
| **scikit-learn** | Logistic Regression, Random Forest |
| **Power BI Desktop** | Interactive 3-page dashboard |

---

## 🔄 Analysis Steps

### Step 1 — Data Loading & Cleaning
- Loaded 7,043 rows from Excel file
- Converted `Total Charges` from text to numeric
- Removed 11 rows with missing values
- Mapped `Senior Citizen` from 0/1 to Yes/No
- Final clean dataset: **7,032 customers**

### Step 2 — Exploratory Analysis (5 Charts)
- Overall churn distribution
- Churn rate by contract type
- Monthly charges vs churn (box plot)
- Customer tenure vs churn (side-by-side histogram)
- Churn rate by internet service type

### Step 3 — Machine Learning Models
- Encoded categorical features using LabelEncoder
- Split data: 80% train / 20% test (random_state=42)
- Trained Logistic Regression and Random Forest
- Evaluated with Accuracy, ROC-AUC, Confusion Matrix
- Generated churn probability scores per customer
- Assigned Risk Level: Low / Medium / High

### Step 4 — Power BI Dashboard
- 3-page interactive dashboard
- Page 1: Churn Overview
- Page 2: High Risk Customers
- Page 3: Retention Strategy

---

## 🔍 Key Findings

| Finding | Detail |
|---|---|
| Overall churn rate | **26.6%** — 1,869 out of 7,032 customers left |
| Highest risk contract | Month-to-month **42.7%** vs Two year **2.8%** |
| Internet service risk | Fiber optic **41.9%** — highest churn type |
| Salary gap | Churned customers pay median **$80/mo** vs **$64/mo** retained |
| Early churn danger | **600 customers** churned in first 3 months |
| Long-term loyalty | Customers with 70+ months tenure almost never leave |
| Tech Support impact | No tech support = **41.7%** churn vs **15.2%** with support |
| Online Security impact | No security = **41.8%** churn vs **14.6%** with security |

---

## 🤖 ML Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Logistic Regression** | **80.5%** | **0.856** |
| Random Forest | 79.2% | 0.831 |

> Logistic Regression outperformed Random Forest — common for telecom churn datasets where feature-churn relationships are largely linear.

### Confusion Matrix (Random Forest)
| | Predicted Retained | Predicted Churned |
|---|---|---|
| **Actually Retained** | 902 ✅ | 110 ❌ |
| **Actually Churned** | 183 ❌ | 212 ✅ |

### Top 10 Churn Predictors
1. 💰 Total Charges (0.205)
2. 💳 Monthly Charges (0.203)
3. 📅 Tenure Months (0.163)
4. 📋 Contract (0.080)
5. 🔒 Online Security (0.056)
6. 💸 Payment Method (0.053)
7. 🛠️ Tech Support (0.038)
8. 👨‍👩‍👧 Dependents (0.037)
9. 💑 Partner (0.027)
10. 💾 Online Backup (0.027)

---

## 💡 Business Recommendations

### Contract Strategy
- ✅ Offer discounts to move month-to-month customers to annual contracts
- ✅ Target new customers (tenure < 3 months) with proactive onboarding calls
- ✅ Send loyalty rewards at month 6 and month 12 milestones

### Pricing
- ✅ Review pricing for customers paying over $80/month
- ✅ Offer bundle discounts to reduce monthly charges
- ✅ Benchmark against competitor pricing for fiber optic plans

### Services
- ✅ Promote Tech Support add-on — reduces churn from 41.7% to 15.2%
- ✅ Promote Online Security — reduces churn from 41.8% to 14.6%
- ✅ Customers with more services have significantly lower churn

### High Risk Customers
- ✅ Use ML model monthly to flag High Risk customers
- ✅ Assign retention team to personally contact all High Risk customers
- ✅ Offer personalised retention deals to customers with 60%+ churn probability

---

## 📊 Dashboard Preview

### Page 1 — Churn Overview
![Churn Overview](Churn-prediction/Screenshots/Churn%20Overview.png)

### Page 2 — High Risk Customers
![High Risk Customers](Churn-prediction/Screenshots/HIgh%20Risk%20Customer.png)

### Page 3 — Retention Strategy
![Retention Strategy](Churn-prediction/Screenshots/Retention%20Strategy.png)

---

## ▶️ How to Run

### 1. Clone the repository
```
git clone https://github.com/PrabinPokhrel/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction
```

### 2. Install required libraries
```
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Download `Telco_customer_churn.xlsx`
- Place it inside the `Churn-prediction/data/` folder

### 4. Run the analysis
```
python Churn-prediction/churn_analysis.py
```

### 5. View the Power BI dashboard
- Open `Churn-prediction/PowerBI/Dashboard.pbix` in Power BI Desktop
- Power BI Desktop is free to download from microsoft.com
- Refresh the data source if prompted

## 👤 Author

**Prabin Pokhrel**
Master's in Business Intelligence — Dalarna University
- GitHub: [@PrabinPokhrel](https://github.com/PrabinPokhrel)

---

*⭐ If you found this project helpful, please give it a star!*

