# Flight Delay Analysis & Prediction

This project applies a full data science pipeline to the **Flight Delay and Cancellation Dataset (2019–2023)**, covering preprocessing, EDA, hypothesis testing, and a complete suite of machine learning models for regression, classification, and clustering.

---

## Project Structure

```
Project/
├── Project codes/          Python modules, Jupyter Notebook, R script
├── Output files/           All generated plots and saved models
│   ├── barplots/
│   ├── boxplots/
│   ├── correlations/
│   ├── cyclical_features/
│   ├── dimensionality_reduction/
│   ├── distributions/
│   ├── heatmaps/
│   ├── lineplots/
│   ├── polar_plots/
│   ├── scatter_plots/
│   ├── violin_plots/
│   └── models/             Per-model plots, saved_models/, saved_models.zip
├── Project datasets/       cleaned_flights.csv, encoding_mappings.csv
├── Project plans/          Project brief (PDF) and structure diagram
├── Project reports/        Final report (PDF)
└── README.md
```

> **Note:** `Project datasets/` is not tracked in git (file size). Run the notebook from the top to generate `cleaned_flights.csv` via the Kaggle download step.

---

## Pipeline Overview

### 1. Data Loading
- Downloads the dataset from Kaggle via `kagglehub`
- Loads ~3 million rows into a pandas DataFrame

### 2. Data Preprocessing & Feature Engineering
- Drops leakage and irrelevant columns; filters cancelled/diverted flights
- Creates new features: month/weekday/season, cyclical time encodings (sin/cos),
  route, average speed, origin/destination state
- Exports `cleaned_flights.csv` (~2.9 M rows, 18 features)

### 3. Data Splitting
- 80/20 train/test split (stratified by year)
- Ordinal/label encoding for categorical features (`ORIGIN_CITY`, `DEST_CITY`,
  `ROUTE`, `ORIGIN_STATE`, `DEST_STATE`)
- StandardScaler on `DISTANCE`, `CRS_ELAPSED_TIME`, `AVG_SPEED`
- Exports `encoding_mappings.csv` for future inference

### 4. Exploratory Data Analysis (EDA)
- Delay distribution, feature distributions, boxplots, correlation heatmap
- Temporal patterns by hour, day, month, season
- Route and city-level delay analysis

Key findings:
- Delays are **right-skewed** with a heavy tail
- Weak linear correlations — non-linear models are more suitable
- Strong dependence on time-of-day, season, and route

### 5. Dimensionality Reduction
- **PCA** (linear): no strong class separation in two components
- **UMAP** (non-linear): reveals hidden clusters and local structure

### 6. Hypothesis Testing
- **H1** (Weekend vs. Weekday): weekends have significantly higher delays
  (Welch t-test, p = 5.06e-16)
- **H2** (Pandemic impact): post-pandemic delays (2022–2023) significantly
  higher than pre-pandemic (2019) (p = 7.95e-158)

---

## Models (Part 2)

All models predict `ARR_DELAY` (arrival delay in minutes).
Regression uses a **winsorized target** (capped at the 99th percentile, ~189 min).
Classification uses **3 classes**: on-time (< 15 min), short delay (15–30 min),
long delay (> 30 min).

### kNN from Scratch (NumPy only)
- Vectorised L2 distance, `k = 5`, balanced class weights, z-score normalisation
- Subsampled to 20 K train / 5 K test (O(n×m) scaling constraint)

### Decision Tree (scikit-learn)
- `max_depth = 12`, `class_weight = "balanced"` (classifier), full ~2.33 M training set

### Random Forest (scikit-learn) — best classifier
- `n_estimators = 100`, `max_depth = 12`, full training set
- **Classification**: Accuracy 56.8%, Weighted F1 0.630
- **Regression**: RMSE 29.83, R² 0.054

### LightGBM — best regressor
- `n_estimators = 200`, `max_depth = 8`, `learning_rate = 0.05`, full training set
- **Classification**: Accuracy 54.5%, Weighted F1 0.614
- **Regression**: RMSE 29.78, R² 0.057

### Deep Learning MLP (PyTorch)
- Architecture: input → 256 → 128 → 64 → output, ReLU + Dropout(0.3)
- Adam, 15 epochs, batch size 1024, 500 K training subsample (CPU constraint)
- **Classification**: Accuracy 54.1%, Weighted F1 0.609
- **Regression**: RMSE 30.01, R² 0.043

### Clustering (unsupervised)
- **KMeans** (k = 2–6, 200 K subsample): best silhouette at k = 3 (score 0.251)
- **DBSCAN** (varying eps, 5 K subsample): density-based, noise identification

---

## Model Saving

All trained models are serialised after training and provided in a zip archive
for future deployment or analysis.

**Location:** `Output files/models/saved_models.zip`

Directory layout inside the zip:

```
saved_models/
├── knn/
│   ├── knn_classification_arrays.npz
│   ├── knn_classification_params.json
│   ├── knn_regression_arrays.npz
│   └── knn_regression_params.json
├── supervised/
│   ├── decision_tree_cls.joblib
│   └── decision_tree_reg.joblib
├── ensemble/
│   ├── lightgbm_classifier.joblib
│   ├── lightgbm_regressor.joblib
│   ├── random_forest_classifier.joblib
│   └── random_forest_regressor.joblib
├── deep_learning/
│   ├── mlp_classifier_config.json
│   ├── mlp_classifier_scaler.joblib
│   ├── mlp_classifier_weights.pt
│   ├── mlp_regressor_config.json
│   ├── mlp_regressor_scaler.joblib
│   └── mlp_regressor_weights.pt
├── clustering/
│   ├── kmeans_k2.joblib … kmeans_k6.joblib
│   └── dbscan_eps*.joblib
├── feature_names.json
└── MODEL_CARD.md
```

`MODEL_CARD.md` documents each model's hyperparameters, training subset,
test-set metrics, and copy-paste loading code.

---

## Running the Project

1. Open `Project codes/main.ipynb` in Jupyter
2. Run all cells top to bottom (**Kernel → Restart & Run All**)
3. The final cell serialises all models and creates `saved_models.zip`

---

## Tech Stack

| Library | Use |
|---|---|
| pandas, NumPy | Data loading, manipulation, KNN implementation |
| scikit-learn | Decision Tree, Random Forest, KMeans, DBSCAN, preprocessing |
| LightGBM | Gradient boosting |
| PyTorch | MLP deep learning |
| UMAP-learn | Non-linear dimensionality reduction |
| Matplotlib, Seaborn | Visualisation |
| KaggleHub | Dataset download |
| joblib, zipfile | Model serialisation |

---

## Authors

- Alexis Barros 2045719
- Vítor Remesso 2050519

Master's Degree in Computer Engineering — University of Madeira

---

## Dataset

Flight Delay and Cancellation Dataset (2019–2023)  
Source: US Department of Transportation, Bureau of Transportation Statistics — via Kaggle
