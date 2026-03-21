# Flight Delay Analysis & Prediction

This project explores flight delay patterns using the **Flight Delay and Cancellation Dataset (2019–2023)**.  
The goal is to understand the factors influencing delays and prepare the data for predictive modeling tasks such as regression, classification, and clustering.

---

## Project Structure
- Project Codes (contains Python, R code and Jupyter Notebook)
- Project Datasets (to be created, contains dataset and aditional csv's)
- Project Plans (information related to the project at hand)
- Project Reports (contains the projects reports)
- README.md

---

## Project Overview

Flight delays are a major operational and economic challenge in aviation.  
This project applies a full **data science pipeline** to analyze and model delay behavior, including:

- Data loading and preprocessing  
- Feature engineering  
- Exploratory Data Analysis (EDA)  
- Dimensionality reduction (PCA & UMAP)  
- Preparation for machine learning models  

The analysis shows that delays are **highly skewed**, weakly explained by simple numerical features, and strongly influenced by **temporal and operational factors** :contentReference[oaicite:0]{index=0}

---

## Pipeline Structure

### 1. Data Loading
- Downloads dataset from Kaggle  
- Loads into a pandas DataFrame  
- Displays preview and summary  

---

### 2. Data Preprocessing & Feature Engineering
- Removes irrelevant and leakage columns  
- Handles missing values  
- Filters cancelled/diverted flights  
- Creates new features:
  - Date features (month, weekday, season)
  - Time features (cyclical encoding)
  - Route and state features  
  - Operational indicators (peak hours)  
  - Average flight speed  
- Produces a clean dataset ready for analysis and modeling  

---

### 3. Data Splitting
- Splits data into training and testing sets  
- Encodes categorical variables  
- Scales numerical features  
- Prevents data leakage by fitting only on training data  

---

### 4. Exploratory Data Analysis (EDA)
- Analyzes delay distribution  
- Visualizes feature distributions and outliers  
- Studies temporal patterns (day, month, season)  
- Examines relationships between variables  
- Generates multiple plots for insights  

Key findings:
- Delays are **right-skewed with extreme values**  
- Weak linear correlations with features  
- Strong dependence on **time, season, and routes**

---

### 5. Dimensionality Reduction (PCA & UMAP)

#### PCA (Linear)
- Captures global variance  
- Shows no strong linear separation between delay classes  

#### UMAP (Non-linear)
- Reveals hidden clusters and local structures  
- Highlights complex relationships in the data  

Insight:
- Flight delays are driven by **non-linear interactions**, suggesting the need for flexible models

---

## Key Insights

- Most flights are on-time or slightly delayed  
- A small subset has **large delays (heavy tail)**  
- Delay patterns vary significantly by:
  - Time of day  
  - Day of week  
  - Season  
  - Route and airline  

- Linear relationships are weak → **non-linear models are more suitable**

---

## Future Work

The next phase includes:

- kNN implementation (from scratch)  
- Supervised models (e.g., Logistic Regression, Random Forest)  
- Ensemble methods (Bagging & Boosting)  
- Deep learning models  
- Clustering (K-Means, DBSCAN)  

---

## Tech Stack

- Python (Pandas, NumPy)  
- Scikit-learn  
- Seaborn & Matplotlib  
- UMAP  
- KaggleHub  

---
## Authors

- Alexis Barros 2045719
- Vítor Remesso 2050519

Master’s Degree in Computer Engineering  
University of Madeira  

---

## Dataset

Flight Delay and Cancellation Dataset (2019–2023)  
Source: Kaggle  

---

## Conclusion

This project establishes a strong analytical foundation by combining preprocessing, EDA, and dimensionality reduction.  

The findings highlight that flight delay behavior is **complex and non-linear**, guiding the use of more advanced models in later stages.

---
