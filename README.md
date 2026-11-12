# Banking Customer Intelligence Platform
### Customer Churn Prediction · Customer Segmentation · Risk Analytics

An end-to-end banking customer analytics project that combines **exploratory data analysis (EDA)**, **supervised machine learning**, **unsupervised customer segmentation**, and an interactive **Streamlit application**.

The platform explores historical customer behaviour, estimates customer churn probability using a trained XGBoost classifier, and assigns customers to behavioural segments using K-Means clustering.

<p align="center">
  <a href="https://github.com/OmarSakibSiam/Banking-Customer-Intelligence-Platform-Churn-Prediction-CustomerSegmentation-RiskAnalytics/blob/main/project.ipynb">Notebook</a> ·
  <a href="https://github.com/OmarSakibSiam/Banking-Customer-Intelligence-Platform-Churn-Prediction-CustomerSegmentation-RiskAnalytics/blob/main/Banking_Customer_Intelligence_EDA_Report.pdf">EDA Report</a> ·
  <a href="https://github.com/OmarSakibSiam/Banking-Customer-Intelligence-Platform-Churn-Prediction-CustomerSegmentation-RiskAnalytics/blob/main/app.py">Streamlit App</a>
</p>

---

## Project at a Glance

| Component | Implementation |
|---|---|
| Dataset | 10,000 banking customer records |
| Target variable | `Exited` (0 = retained, 1 = churned) |
| Exploratory analysis | Pandas, Matplotlib, Seaborn |
| Preprocessing | ColumnTransformer, OneHotEncoder |
| Churn prediction | Random Forest and XGBoost |
| Model selection | GridSearchCV with 5-fold cross-validation |
| Customer segmentation | K-Means clustering |
| Clustering features | Age, Balance, NumOfProducts, IsActiveMember, HasZeroBalance |
| Dimensionality reduction | PCA for cluster visualization |
| Model persistence | Joblib |
| Interactive interface | Streamlit |

## Repository Contents

The following project files are available directly in this repository:

| File | What it contains |
|---|---|
| [`project.ipynb`](./project.ipynb) | Main Jupyter notebook: data inspection, EDA, feature engineering, preprocessing, model training and evaluation, clustering, visualizations, and model-artifact preparation. |
| [`Banking_Customer_Intelligence_EDA_Report.pdf`](./Banking_Customer_Intelligence_EDA_Report.pdf) | PDF report with the EDA charts, analysis questions, interpretations, findings, and supplementary customer-segmentation profiles. |
| [`app.py`](./app.py) | Streamlit application source code for individual customer analysis, churn prediction, and cluster-profile exploration. |
| [`banking_customer_app.pkl`](./banking_customer_app.pkl) | Serialized model bundle loaded by the Streamlit application. It contains the fitted preprocessing and model objects plus supporting metadata. |
| [`Churn_Modelling.csv`](./Churn_Modelling.csv) | Banking customer dataset used in the project. |
| [`requirments.txt`](./requirments.txt) | Python package versions recorded for the project environment. |

## 1. Business Problem

Customer churn is an important concern for banks because losing existing customers can affect revenue, customer lifetime value, and long-term relationships.

This project explores the following questions:

- Which customer characteristics and behaviours are associated with historical churn?
- Do churn patterns differ across geography, age groups, product usage, and activity status?
- Can a machine-learning model estimate the probability that a customer will leave?
- Can customers be grouped into interpretable segments based on their banking behaviour?
- How can these model outputs be made accessible through an interactive application?

The goal is to provide complementary views of customer behaviour: **churn prediction** at the individual-customer level and **segmentation** at the customer-group level.

## 2. Project Workflow

```text
             Churn_Modelling.csv
                      |
                      v
           Data Inspection and EDA
                      |
                      v
              Feature Engineering
                      |
                      v
        Categorical / Numeric Preprocessing
                      |
             +--------+--------+
             |                 |
             v                 v
       Churn Prediction   Customer Segmentation
             |                 |
             v                 v
      Random Forest       Feature Selection
        & XGBoost                |
             |                  v
             v             StandardScaler
       GridSearchCV              |
             |                  v
             v              K-Means (3)
      Model Evaluation           |
             |                  v
             |             PCA Visualization
             |                  |
             +--------+---------+
                      |
                      v
          Save fitted objects with Joblib
                      |
                      v
          banking_customer_app.pkl
                      |
                      v
                Streamlit App
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Churn Probability       Customer Segment
```

## 3. Exploratory Data Analysis

The dataset contains **10,000 rows and 14 original columns**. The EDA checks dataset dimensions, data types, missing values, descriptive statistics, duplicate rows, and duplicate customer IDs before examining customer demographics and banking behaviour.

The EDA report includes analysis of:

- Customer distribution by geography and gender
- Age distribution and churn across age groups
- Account balance and estimated salary
- Credit-score distribution and categories
- Customer tenure and number of products
- Credit-card ownership and active-member status
- Overall churn distribution
- Correlation between selected numerical features and churn
- Retained-versus-churned customer profiles
- A historical customer value/risk matrix

### Selected EDA Findings

| Observation | Result |
|---|---:|
| Retained customers | 7,963 (79.63%) |
| Churned customers | 2,037 (20.37%) |
| Customers with zero balance | 3,617 (36.17%) |
| Germany observed churn rate | 32.44% |
| France observed churn rate | 16.15% |
| Spain observed churn rate | 16.67% |
| Inactive-member observed churn rate | 26.85% |
| Active-member observed churn rate | 14.27% |

Age has the strongest positive correlation with `Exited` among the selected numerical features in the EDA correlation analysis (0.29). The 46–55 and 56–65 age groups show the highest observed churn rates in the age-band analysis.

These figures describe patterns in this dataset. They should not be interpreted as evidence that any single feature causes churn.

**Read the full analysis:** [Banking Customer Intelligence — EDA Report (PDF)](./Banking_Customer_Intelligence_EDA_Report.pdf)

## 4. Feature Engineering and Preprocessing

The notebook prepares the data for modelling through the following steps:

1. Separates `Exited` as the target variable.
2. Removes identifier/non-predictive columns: `RowNumber`, `CustomerId`, and `Surname`.
3. Creates `HasZeroBalance` to identify customers whose account balance is zero.
4. Applies a `ColumnTransformer` with `OneHotEncoder` to the categorical columns `Geography` and `Gender`.
5. Preserves the remaining numerical and binary features through the preprocessing pipeline.
6. Splits the transformed data into training and testing subsets using a 70/30 split and `random_state=42`.

The fitted transformer is saved with the model objects so the Streamlit application can apply the same preprocessing to new customer inputs.

## 5. Churn Prediction

Two supervised classification algorithms are explored:

### Random Forest

- Algorithm: `RandomForestClassifier`
- Hyperparameter tuning: `GridSearchCV`
- Cross-validation: 5 folds
- Search scoring metric: F1 score

### XGBoost

- Algorithm: `XGBClassifier`
- Hyperparameter tuning: `GridSearchCV`
- Cross-validation: 5 folds
- Search scoring metric: F1 score

The notebook compares the models using its evaluation outputs. The fitted XGBoost model is the classifier loaded by the Streamlit application.

The application obtains the positive-class probability through `predict_proba()` and applies the decision threshold saved in the model artifact (falling back to 0.5 if no threshold is stored). It displays the estimated churn probability and the corresponding prediction.

> **Evaluation note:** The EDA PDF is specifically an exploratory-analysis report; it does not report predictive-model results. Consult the modelling and evaluation sections of [`project.ipynb`](./project.ipynb) for the model experiment outputs. This README does not present unverified performance figures.

## 6. Customer Segmentation

K-Means clustering is used to identify groups of customers with similar characteristics.

### Features used

- `Age`
- `Balance`
- `NumOfProducts`
- `IsActiveMember`
- `HasZeroBalance`

The selected features are standardized with `StandardScaler`. The notebook examines candidate cluster counts and fits a final **3-cluster K-Means model**. PCA is used to visualize the clusters in two dimensions.

### Customer Segment Profiles

The supplementary segmentation analysis reports the following profiles:

| Cluster | Segment | Customers | Share | Historical Churn Rate |
|---|---|---:|---:|---:|
| 0 | Active Balance-Holding Customers | 3,278 | 32.78% | 16.93% |
| 1 | Inactive Balance-Holding Customers | 3,104 | 31.04% | 31.60% |
| 2 | Predominantly Zero-Balance Customers | 3,618 | 36.18% | 13.85% |

**Interpretation:**

- **Cluster 0 — Active Balance-Holding:** All customers in this cluster are active members. Their average balance is 119,229.50 and they hold 1.38 products on average.
- **Cluster 1 — Inactive Balance-Holding:** Customers are inactive members despite having substantial balances. This cluster has the highest historical churn rate (31.60%).
- **Cluster 2 — Predominantly Zero-Balance:** Almost all customers have a zero balance (99.97%). This is the largest cluster and has the lowest historical churn rate (13.85%).

**Important distinction:** `Exited` is not used to create the K-Means clusters. Historical churn rates are calculated after clustering to describe the outcomes within each segment. They are not individual churn predictions. Cluster IDs are arbitrary labels.

## 7. Streamlit Application

The interactive application is implemented in [`app.py`](./app.py). It loads the serialized artifact [`banking_customer_app.pkl`](./banking_customer_app.pkl) using Joblib.

### Application pages

| Page | Functionality |
|---|---|
| **Customer Analyzer** | Accepts customer details and returns an estimated churn probability, churn classification, K-Means segment, and the segment's profile. |
| **Cluster Profiles** | Displays the saved cluster-profile table and lets users explore an individual segment. |
| **About the Model** | Explains the churn-prediction and customer-segmentation models and their interpretation. |

### How an individual customer is analyzed

1. The user enters credit score, geography, gender, age, tenure, balance, number of products, credit-card status, active-member status, and estimated salary.
2. The application derives `HasZeroBalance` from the entered balance.
3. The saved `ColumnTransformer` transforms the raw customer input.
4. The transformed features are passed to the saved XGBoost classifier to calculate churn probability and the predicted class.
5. The clustering features are selected and transformed with the saved `StandardScaler`.
6. The fitted K-Means model assigns the customer to a cluster.
7. The application displays the churn result and the corresponding segment profile.

The application does **not** retrain the models when a user submits a customer. It uses the fitted objects stored in the Joblib artifact.

### Application preview / deployment

The repository contains the Streamlit source code and model artifact. A public Streamlit deployment URL was not present in the repository files inspected for this README. If the app is deployed on Streamlit Community Cloud, add its public URL here:

**Live App:** https://bankingintelligence.streamlit.app/

## 8. Technology Stack

| Category | Libraries / Tools |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Gradient boosting | XGBoost |
| Preprocessing | ColumnTransformer, OneHotEncoder, StandardScaler |
| Classification | Random Forest, XGBoost |
| Clustering | K-Means |
| Dimensionality reduction | PCA |
| Hyperparameter tuning | GridSearchCV |
| Model serialization | Joblib |
| Interactive application | Streamlit |
| Development environment | Jupyter Notebook |

## 9. Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/OmarSakibSiam/Banking-Customer-Intelligence-Platform-Churn-Prediction-CustomerSegmentation-RiskAnalytics.git
cd Banking-Customer-Intelligence-Platform-Churn-Prediction-CustomerSegmentation-RiskAnalytics
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

The dependency file in the repository is currently named `requirments.txt`:

```bash
pip install -r requirments.txt
```

The current dependency file does not list Streamlit, although `app.py` imports it. Install Streamlit separately before launching the app:

```bash
pip install streamlit
```

### 4. Launch the application

Make sure `app.py` and `banking_customer_app.pkl` are in the same directory, then run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, usually `http://localhost:8501`.

> **Environment compatibility:** The model artifact was serialized with Joblib and depends on compatible versions of Python and its machine-learning libraries. If the artifact fails to load, use the compatible package versions from the environment in which it was created and regenerate the artifact if necessary.

## 10. Limitations and Responsible Use

- The analysis is based on the supplied 10,000-record dataset and may not generalize to other banks or populations.
- EDA group differences and correlations are descriptive, not causal.
- The value/risk matrix in the EDA uses median account balance as a simple value proxy and the already-observed `Exited` outcome as risk status; it is retrospective, not a future-risk model.
- Cluster churn rates describe historical outcomes within groups and must not be confused with an individual customer's predicted churn probability.
- Model predictions are estimates and should not be treated as guarantees or used for real banking decisions without further validation, monitoring, fairness assessment, and governance.

---

## Author

**Omar Sakib Siam**

- [GitHub Profile](https://github.com/OmarSakibSiam)
- [Project Repository](https://github.com/OmarSakibSiam/Banking-Customer-Intelligence-Platform-Churn-Prediction-CustomerSegmentation-RiskAnalytics)
