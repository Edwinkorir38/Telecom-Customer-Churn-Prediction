<h1 align="center"> Telecom Customer Churn Prediction</h1>

<p align="center">
  <img src="images/stock-photo-customer-churn-is-shown-using-a-text-2443621059.jpg" alt="Telecom Churn Banner" width="80%"/>
</p>

<div align="center" style="margin-top: 20px;">

![GitHub forks](https://img.shields.io/github/forks/Edwinkorir38/Telecom-Customer-Churn-Prediction?style=flat-square&logo=github)
![GitHub Repo stars](https://img.shields.io/github/stars/Edwinkorir38/Telecom-Customer-Churn-Prediction?logo=GitHub)
![GitHub contributors](https://img.shields.io/github/contributors/Edwinkorir38/Telecom-Customer-Churn-Prediction?logo=github&color=blue)
[![View Demo](https://img.shields.io/badge/View%20Demo-orange?style=flat-square&logo=github&logoColor=black&labelColor=hex)](https://churnprediction.pythonanywhere.com/)
[![MIT License](https://img.shields.io/badge/MIT%20License-blue?style=flat-square&logo=Github&logoColor=black)](https://github.com/Edwinkorir38/Telecom-Customer-Churn-Prediction/blob/main/LICENSE)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Edwinkorir38/Telecom-Customer-Churn-Prediction?style=flat-square&logo=github&logoColor=black)
![GitHub issues](https://img.shields.io/github/issues/Edwinkorir38/Telecom-Customer-Churn-Prediction?logo=github&logoColor=black)

</div>

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Data Understanding](#-data-understanding)
- [Project Structure](#️-project-structure)
- [Methodology](#-methodology)
- [Model Results](#-model-results)
- [Key Findings & Recommendations](#-key-findings--recommendations)
- [How to Install and Run](#️-how-to-install-and-run-this-project)
- [How to Use](#-how-to-use-this-project)
- [Deployment](#-deployment)
- [License](#-license)
- [Feedback and Support](#-feedback-and-support)

---

##  Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Telecommunications |
| **Problem Type** | Binary Classification (Supervised Learning) |
| **Target Variable** | `Churn` (True = Customer left, False = Customer retained) |
| **Dataset** | Telecom customer records — Train: 3,333 rows · Test: 1,667 rows |
| **Features** | 19 input features covering demographics, plan subscriptions, and usage behaviour |
| **Models Evaluated** | Logistic Regression, Decision Tree, Random Forest, SVM (Linear), SVM (RBF) |
| **Chosen Model** |  Random Forest Classifier |
| **Deployment Artifacts** | `random_forest_model.pkl`, `label_encoder.pkl`, `standard_scaler.pkl` |
| **Web App** | Flask application deployed on PythonAnywhere |

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs **5–25× more** than retaining an existing one. This project builds a complete end-to-end machine learning pipeline that identifies at-risk customers **before** they leave, enabling the business to intervene proactively and minimise revenue loss.

---

##  Business Problem
![image](images/freepik__visual-of-telecom-churn-rate-trend-declining-custo__63267.png)
Telecom companies operate in a highly competitive market where customers can switch providers with minimal friction. **Churn** — when a customer terminates their subscription or switches to a competitor — directly erodes revenue and increases operational costs through:

-  **Lost recurring revenue** from the departing customer.
-  **Increased acquisition costs** to replace churned customers with new ones.
-  **Negative word-of-mouth** effects from dissatisfied departing customers.

### The Core Business Question
![image](images/customer-loyalty-service-efficiency-strategy-concept.jpg)
> *"Which of our current customers are most likely to leave in the near future, and what can we do to retain them?"*

### Why Machine Learning?

Traditional rule-based approaches (e.g. *"flag customers with 3+ support calls"*) are too rigid and miss complex interaction effects between features. A trained ML model learns subtle, multi-dimensional patterns — such as a customer on an international plan who is a heavy daytime caller and has contacted support twice — that a human analyst would overlook.

### Business Value

- **Targeted retention campaigns** — focus resources on high-risk customers rather than the entire base.
- **Cost savings** — reducing churn by even 1–2% can represent millions in preserved annual recurring revenue.
- **Improved customer satisfaction** — proactively addressing issues (billing, plan suitability) before a customer decides to leave.

---

##  Project Objectives

### Primary Objective
Build a **predictive classification model** that accurately identifies telecom customers who are likely to churn, enabling the business to take preemptive retention action.

### Secondary Objectives
1. **Understand churn drivers** — identify which customer attributes and usage patterns are most strongly associated with churn through EDA and feature importance analysis.
2. **Benchmark multiple algorithms** — evaluate five different classification approaches to select the most effective model empirically.
3. **Ensure deployability** — produce serialised, production-ready model artifacts (model + preprocessing transformers) that integrate into a customer scoring pipeline.
4. **Handle class imbalance** — recognise and account for the ~85/15 class distribution to avoid a model that is superficially accurate but poor at detecting actual churners.

### Success Criteria

| Metric | Minimum Target | Rationale |
|--------|---------------|-----------|
| **Recall (Churn class)** | ≥ 70% | Catching churners is the primary goal |
| **Precision (Churn class)** | ≥ 60% | Avoid over-targeting non-churners with costly offers |
| **ROC-AUC** | ≥ 0.85 | Overall discriminative power across thresholds |
| **F1-Score (Churn class)** | ≥ 0.65 | Balanced performance on the minority class |

---

##  Dataset

Download the dataset for custom training from the [data folder](data/).

The dataset is a well-known **telecom customer churn dataset** (commonly associated with the Kaggle / UCI Churn in Telecoms dataset). It contains customer-level records from a US telecom operator, pre-split into training and test sets.

| Split | Rows | Columns | Churn Rate |
|-------|------|---------|------------|
| **Train** | 3,333 | 20 | ~14.5% |
| **Test** | 1,667 | 20 | ~14.5% |

>  **Class Imbalance Note:** ~85.5% of customers did **not** churn vs ~14.5% who did. Raw accuracy is therefore a misleading metric — precision, recall, F1-score, and AUC-ROC are the primary evaluation metrics in this project.

---

##  Data Understanding

### Feature Dictionary

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `State` | Categorical | US state of the customer — *dropped in preprocessing* |
| 2 | `Account length` | Numerical | Days the customer has been with the company |
| 3 | `Area code` | Categorical | Phone area code (408, 415, 510) — *dropped in preprocessing* |
| 4 | `International plan` | Binary | Whether the customer has an international calling plan (Yes/No) |
| 5 | `Voice mail plan` | Binary | Whether the customer has a voicemail plan (Yes/No) |
| 6 | `Number vmail messages` | Numerical | Number of voicemail messages received |
| 7 | `Total day minutes` | Numerical | Total minutes of daytime calls |
| 8 | `Total day calls` | Numerical | Total number of daytime calls |
| 9 | `Total day charge` | Numerical | Daytime call charge — *dropped* (perfectly correlated with minutes) |
| 10 | `Total eve minutes` | Numerical | Total minutes of evening calls |
| 11 | `Total eve calls` | Numerical | Total number of evening calls |
| 12 | `Total eve charge` | Numerical | Evening call charge — *dropped* (perfectly correlated with minutes) |
| 13 | `Total night minutes` | Numerical | Total minutes of night calls |
| 14 | `Total night calls` | Numerical | Total number of night calls |
| 15 | `Total night charge` | Numerical | Night call charge — *dropped* (perfectly correlated with minutes) |
| 16 | `Total intl minutes` | Numerical | Total minutes of international calls |
| 17 | `Total intl calls` | Numerical | Total number of international calls |
| 18 | `Total intl charge` | Numerical | International call charge — *dropped* (perfectly correlated with minutes) |
| 19 | `Customer service calls` | Numerical | Number of calls made to customer service |
| 20 | `Churn` | Boolean (Target) | Whether the customer churned: True / False |

### Data Quality Summary

| Check | Result |
|-------|--------|
| Missing values |  Zero — no imputation required |
| Duplicate rows |  None detected |
| Class balance |  Imbalanced — 85.5% vs 14.5% |
| Multicollinearity |  4 charge columns are linear derivatives of minute columns — dropped |
| Schema consistency (train vs test) |  Identical structure and data types |

### Known Churn Risk Factors (Domain Knowledge)

Based on industry research, the following features are expected to be the strongest predictors:

-  **Customer service calls** — high call frequency signals unresolved complaints.
-  **International plan** — subscribers dissatisfied with international rates are more likely to switch.
-  **Total day minutes/charge** — heavy daytime users may be price-sensitive and actively seeking cheaper alternatives.
-  **Voice mail plan** — customers with value-added services tend to be more loyal (lower expected churn).

---

##  Project Structure

```
Telecom-Customer-Churn-Prediction/
│
├── data/
│   ├── raw/                  # Original train.csv and test.csv
│   └── processed/            # Preprocessed train_final and test_final
│
├── notebooks/
│   └── Telecom-customer-churn-prediction.ipynb   # Main analysis notebook
│
├── models/
│   ├── random_forest_model.pkl   # Trained Random Forest (chosen model)
│   ├── label_encoder.pkl         # Fitted LabelEncoder for categorical features
│   └── standard_scaler.pkl       # Fitted StandardScaler for numerical features
│
├── reports/                  # Power BI and other analysis reports
├── images/                   # Images used in documentation and the web app
├── static/                   # CSS and static assets for the Flask app
├── templates/                # HTML templates for Flask web pages
├── app.py                    # Main Flask web application
└── requirements.txt          # Python dependencies
```

---

##  Methodology

The project follows a structured data science pipeline:

### 1.  Data Collection & Loading
Train and test datasets are loaded from CSV files. The test set is kept fully separate throughout model development to ensure unbiased final evaluation.

### 2.  Exploratory Data Analysis (EDA)
- **Churn distribution** — donut chart reveals the ~85/15 class imbalance.
- **Pairplot** — confirms perfect multicollinearity between minute and charge column pairs.
- **Correlation matrix** — Pearson heatmap on preprocessed features; `Customer service calls` and `International plan` show the strongest positive correlation with churn.
- **PCA scatter** — 2D principal component visualisation of class separability (for exploration only, not used in modelling).

### 3.  Data Preprocessing
| Step | Detail |
|------|--------|
| **Drop columns** | `State`, `Area code`, and all 4 charge columns removed |
| **Label encoding** | Binary categoricals (`International plan`, `Voice mail plan`) encoded as 0/1 |
| **Standard scaling** | All numerical features scaled to mean=0, std=1 |
| **Save transformers** | `LabelEncoder` and `StandardScaler` serialised for deployment |

### 4.  Model Building
Five classifiers are trained and evaluated with a consistent evaluation wrapper:

| Model | Key Hyperparameters | Notes |
|-------|-------------------|-------|
| **Logistic Regression** | C=1.0, penalty=L2, solver=liblinear | Linear baseline |
| **Decision Tree** | max_depth=9, criterion=gini | Interpretable; prone to overfitting |
| **Random Forest** | n_estimators=100, max_depth=9, criterion=gini |  Selected for deployment |
| **SVM (Linear)** | C=1.0, kernel=linear | Linear boundary; sensitive to scaling |
| **SVM (RBF)** | C=10.0, kernel=rbf, gamma=0.1 | Non-linear; computationally heavier |

### 5.  Model Evaluation
Each model is assessed using:
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- ROC curve & AUC score
- Precision-Recall curve & Average Precision
- Cohen's Kappa score
- Performance comparison bar charts (training set vs. held-out test set)

---

##  Model Results

### Why Random Forest Was Selected

The **Random Forest Classifier** was chosen for deployment based on consistently superior performance across all key metrics on both the validation split and the held-out test set:

- Its ensemble approach (averaging 100 diverse decision trees) substantially reduces variance compared to a single Decision Tree.
- It handles the non-linear decision boundaries in this dataset better than linear models.
- It is robust to overfitting even without aggressive hyperparameter tuning.
- Expected test AUC: **0.88 – 0.93**

### Expected Model Ranking (Test Set)

| Rank | Model | Strengths |
|------|-------|-----------|
|  1 | **Random Forest** | Best AUC, F1, and recall balance |
|  2 | **SVM (RBF)** | Strong non-linear separation |
|  3 | **Decision Tree** | Interpretable but slightly overfits |
| 4 | **SVM (Linear)** | Stable but limited by linear boundary |
| 5 | **Logistic Regression** | Good baseline; lower recall on churn class |

---

##  Key Findings & Recommendations

### Key Findings
- **~14.5% churn rate** — a classic imbalanced classification problem where accuracy alone is misleading.
- **Customer service calls** is the single strongest churn signal — customers who call support frequently are far more likely to churn.
- **International plan subscribers** churn at a significantly higher rate — pricing dissatisfaction is a likely driver.
- **Charge columns are redundant** — all four are linear derivatives of their corresponding minute columns and were removed to eliminate multicollinearity.

### Recommendations for Improvement

1. **Handle class imbalance** — apply `class_weight='balanced'` in the Random Forest or use SMOTE oversampling on the training set to improve recall on the minority (churn) class.
2. **Tune the decision threshold** — the default 0.5 probability cutoff may not be optimal. Use a cost-sensitive threshold derived from the business's customer acquisition vs. retention cost ratio.
3. **Hyperparameter optimisation** — use `RandomizedSearchCV` to tune `n_estimators`, `max_depth`, and `max_features`.
4. **Feature engineering** — create new signals such as call-to-charge ratios, rolling usage trends, or tenure segments.
5. **Retain geographic signal** — instead of dropping `State` entirely, consider grouping states into regions or using target encoding to capture regional churn patterns.
6. **Monitor model drift** — once deployed, track prediction distributions and retrain periodically as customer behaviour evolves.
7. **Explainability** — implement SHAP values to explain individual predictions for retention teams acting on model outputs.

---

##  How to Install and Run this Project

### Prerequisites
- Python 3.8+
- pip

### Steps

**1. Clone the repository:**
```bash
git clone https://github.com/Edwinkorir38/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Explore the project structure** to familiarise yourself with the notebooks, data, and models.

**4. Run the web application:**
```bash
python app.py
```

---

##  How to Use this Project

Once the app is running:

**1.** Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

**2.** You will see a user-friendly web interface. Explore the available features and prediction options.

![web-interface](images/webapp_on_local_machine.png)

**3.** Input customer data fields (plan details, call usage, service history) and click **Predict** to get a churn probability score.

**4.** The app will return a prediction result — churned or not churned — along with a confidence score.

![prediction-results](images/local_machine_after_prediction.png)

**5.** Based on the results, you can take targeted actions such as:
- Offering a discounted plan upgrade to high-risk customers.
- Routing frequent service callers to a dedicated retention team.
- Applying loyalty incentives to international plan subscribers showing churn signals.

---

##  Deployment

The project is deployed as a **Flask web application**. To deploy to your own hosting platform:

**1.** Choose a hosting platform — popular options include Heroku, AWS, Azure, or PythonAnywhere.

**2.** Set up an account if you don't already have one.

**3.** Prepare your project — ensure `requirements.txt` is up to date and any environment variables (e.g. secret keys) are configured per the platform's guidelines.

**4.** Deploy using the platform's provided tools or CLI.

**5.** Once live, access your app via the URL provided by the hosting platform.

For detailed deployment instructions, refer to your chosen platform's official documentation.

<p align="center">
  <a href="https://telecom-customer-churn-prediction-1-9cn5.onrender.com/" style="color:#FF5733; font-size: 18px;">
     View Live Deployed Web App
  </a>
</p>

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open source license that allows you to use, modify, and distribute this project for both commercial and non-commercial purposes.

---

##  Feedback and Support

If you have any feedback, suggestions, or questions regarding the project, please [create an issue](https://github.com/Edwinkorir38/Telecom-Customer-Churn-Prediction/issues) in the repository or contact me directly at:

##  Contact

| | |
|---|---|
| **Name** | Edwin Korir |
| **Email** | [ekorir99@gmail.com](mailto:ekorir99@gmail.com) |
| **GitHub** | [github.com/Edwinkorir38](https://github.com/Edwinkorir38) |
| **LinkedIn** | [linkedin.com/in/edwin-korir-90a794382](https://linkedin.com/in/edwin-korir-90a794382) |
---

<p align="center">
  <strong>If you find this repository helpful, please consider giving it a  star!</strong><br/>
  Your support helps others discover this work and motivates continued improvement. Thank you! 
</p>

<p align="center">Happy analysing and predicting </p>

