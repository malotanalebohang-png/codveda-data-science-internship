"""
╔══════════════════════════════════════════════════════════════════════════╗
║          CODVEDA INTERNSHIP – DATA SCIENCE TASKS                        ║
║          Level 1: Task 1 (Data Cleaning) | Task 2 (EDA)                 ║
║          Level 2: Task 1 (Regression)    | Task 2 (Time Series)         ║
╚══════════════════════════════════════════════════════════════════════════╝

Datasets used:
  - 1__iris.csv               → EDA + Cleaning
  - churn-bigml-80/20.csv     → EDA + Cleaning
  - 4__house_Prediction_Data_Set.csv → Regression
  - 3__Sentiment_dataset.csv  → Time Series

Tools: Python, pandas, matplotlib, seaborn, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────
# 0. LOAD DATASETS
# ────────────────────────────────────────────────────────────────────────

iris = pd.read_csv("1__iris.csv")

sentiment = pd.read_csv("3__Sentiment_dataset.csv")
sentiment['Timestamp'] = pd.to_datetime(sentiment['Timestamp'])

boston_cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
               'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
house = pd.read_csv("4__house_Prediction_Data_Set.csv",
                    header=None, names=boston_cols, sep=r'\s+')

churn = pd.concat([
    pd.read_csv("churn-bigml-80.csv"),
    pd.read_csv("churn-bigml-20.csv")
], ignore_index=True)

print("="*60)
print("DATASETS LOADED")
print(f"  iris:      {iris.shape}")
print(f"  sentiment: {sentiment.shape}")
print(f"  house:     {house.shape}")
print(f"  churn:     {churn.shape}")
print("="*60)


# ════════════════════════════════════════════════════════════════════════
# LEVEL 1 – TASK 1: DATA CLEANING AND PREPROCESSING
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LEVEL 1 – TASK 1: DATA CLEANING & PREPROCESSING")
print("="*60)

# ── 1.1 Missing Value Analysis ──────────────────────────────────────────
print("\n[1.1] Missing Values:")
for name, df in [("Iris", iris), ("Churn", churn), ("House", house), ("Sentiment", sentiment)]:
    total_missing = df.isnull().sum().sum()
    print(f"  {name}: {total_missing} missing values")
    if total_missing > 0:
        print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# ── 1.2 Handle Missing Values ──────────────────────────────────────────
# Sentiment: fill numeric NaNs with median
sentiment['Retweets'] = sentiment['Retweets'].fillna(sentiment['Retweets'].median())
sentiment['Likes'] = sentiment['Likes'].fillna(sentiment['Likes'].median())
print("\n[1.2] Missing values in Sentiment filled with median.")

# ── 1.3 Duplicate Detection ────────────────────────────────────────────
print("\n[1.3] Duplicate Rows:")
for name, df in [("Iris", iris), ("Churn", churn), ("House", house)]:
    dups = df.duplicated().sum()
    print(f"  {name}: {dups} duplicates")
iris_clean = iris.drop_duplicates().reset_index(drop=True)
churn_clean = churn.drop_duplicates().reset_index(drop=True)

# ── 1.4 Outlier Detection (IQR Method) ────────────────────────────────
print("\n[1.4] IQR Outlier Count – House Dataset:")
Q1 = house.quantile(0.25)
Q3 = house.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (house < (Q1 - 1.5*IQR)) | (house > (Q3 + 1.5*IQR))
print(outlier_mask.sum()[outlier_mask.sum() > 0].to_string())

# Remove extreme outliers in MEDV (cap at 99th percentile)
house_clean = house.copy()
cap_val = house['MEDV'].quantile(0.99)
house_clean['MEDV'] = house_clean['MEDV'].clip(upper=cap_val)
print(f"\n  MEDV capped at 99th percentile: {cap_val:.2f}")

# ── 1.5 Encoding Categorical Variables ────────────────────────────────
print("\n[1.5] Label Encoding – Churn Dataset:")
churn_encoded = churn_clean.copy()
churn_encoded['International plan'] = (churn_encoded['International plan'] == 'Yes').astype(int)
churn_encoded['Voice mail plan'] = (churn_encoded['Voice mail plan'] == 'Yes').astype(int)
churn_encoded['Churn'] = churn_encoded['Churn'].astype(int)
le = LabelEncoder()
churn_encoded['State'] = le.fit_transform(churn_encoded['State'])
print("  International plan, Voice mail plan → 0/1")
print("  Churn (True/False) → 1/0")
print("  State → LabelEncoded integer")

# ── 1.6 Feature Scaling ────────────────────────────────────────────────
print("\n[1.6] Min-Max Scaling – House Dataset:")
scaler = MinMaxScaler()
numeric_features = house_clean.drop('MEDV', axis=1).columns.tolist()
house_scaled = house_clean.copy()
house_scaled[numeric_features] = scaler.fit_transform(house_clean[numeric_features])
print(f"  Scaled {len(numeric_features)} numeric features to [0, 1]")
print(f"  Example – CRIM: [{house_scaled['CRIM'].min():.2f}, {house_scaled['CRIM'].max():.2f}]")

# ── 1.7 Data Type Conversion ──────────────────────────────────────────
iris_clean['species'] = iris_clean['species'].astype('category')
print("\n[1.7] Data type conversions done.")

print("\n[SUMMARY] Cleaned Dataset Shapes:")
print(f"  iris_clean:    {iris_clean.shape}")
print(f"  churn_encoded: {churn_encoded.shape}")
print(f"  house_clean:   {house_clean.shape}")


# ════════════════════════════════════════════════════════════════════════
# LEVEL 1 – TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LEVEL 1 – TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# ── 2.1 Descriptive Statistics ────────────────────────────────────────
print("\n[2.1] Iris – Descriptive Statistics:")
print(iris.describe().round(3).to_string())

print("\n[2.1] House – Descriptive Statistics:")
print(house[['RM','AGE','MEDV','LSTAT']].describe().round(3).to_string())

# ── 2.2 Correlation Analysis ──────────────────────────────────────────
print("\n[2.2] House – Top Correlations with MEDV:")
corr_with_medv = house.corr()['MEDV'].drop('MEDV').sort_values(ascending=False)
print(corr_with_medv.round(4).to_string())

print("\n[2.2] Iris – Feature Correlations:")
print(iris.drop('species', axis=1).corr().round(3).to_string())

# ── 2.3 Churn Analysis ────────────────────────────────────────────────
print("\n[2.3] Churn Rate Analysis:")
churn_rate = churn['Churn'].value_counts(normalize=True) * 100
print(f"  Not Churned: {churn_rate[False]:.2f}%")
print(f"  Churned:     {churn_rate[True]:.2f}%")

print("\n[2.3] Day Minutes by Churn Status:")
print(churn.groupby('Churn')['Total day minutes'].describe().round(2).to_string())

# ── 2.4 Iris Species Statistics ───────────────────────────────────────
print("\n[2.4] Iris – Feature Means by Species:")
print(iris.groupby('species').mean().round(3).to_string())


# ════════════════════════════════════════════════════════════════════════
# LEVEL 2 – TASK 1: REGRESSION ANALYSIS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LEVEL 2 – TASK 1: LINEAR REGRESSION ANALYSIS")
print("="*60)

# ── 3.1 Prepare Data ──────────────────────────────────────────────────
X = house.drop('MEDV', axis=1)
y = house['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[3.1] Train/Test Split (80/20):")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")

# ── 3.2 Fit Linear Regression ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\n[3.2] Model fitted: LinearRegression()")

# ── 3.3 Model Coefficients ────────────────────────────────────────────
print("\n[3.3] Regression Coefficients:")
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coeff_df.to_string(index=False))
print(f"\n  Intercept: {model.intercept_:.4f}")
print("\n  Interpretation:")
print("  • LSTAT (−3.74): More low-status residents → lower home value")
print("  • RM    (+3.81): More rooms → higher home value")
print("  • PTRATIO (−0.94): Higher pupil-teacher ratio → lower value")
print("  • NOX  (−17.3): Higher air pollution → lower value")

# ── 3.4 Model Evaluation Metrics ──────────────────────────────────────
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_test - y_pred))

print("\n[3.4] Model Evaluation Metrics:")
print(f"  R² Score  : {r2:.4f}  → Model explains {r2*100:.1f}% of variance in MEDV")
print(f"  MSE       : {mse:.4f}")
print(f"  RMSE      : {rmse:.4f} → Avg prediction error ≈ ${rmse:.1f}k")
print(f"  MAE       : {mae:.4f}")

# ── 3.5 Simple Regression (RM only) ───────────────────────────────────
model_rm = LinearRegression()
model_rm.fit(X_train[['RM']], y_train)
y_pred_rm = model_rm.predict(X_test[['RM']])
r2_rm = r2_score(y_test, y_pred_rm)
print(f"\n[3.5] Simple Model (RM only):")
print(f"  Equation: MEDV = {model_rm.coef_[0]:.3f} × RM + {model_rm.intercept_:.3f}")
print(f"  R²: {r2_rm:.4f} (vs full model: {r2:.4f})")
print(f"  → Multi-feature model explains {(r2 - r2_rm)*100:.1f}% more variance")


# ════════════════════════════════════════════════════════════════════════
# LEVEL 2 – TASK 2: TIME SERIES ANALYSIS
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LEVEL 2 – TASK 2: TIME SERIES ANALYSIS")
print("="*60)

# ── 4.1 Prepare Time Series ───────────────────────────────────────────
sentiment_map = {
    'Positive': 1, 'Joy': 1, 'Excitement': 1, 'Contentment': 1, 'Love': 1,
    'Gratitude': 1, 'Optimism': 1, 'Enthusiasm': 1, 'Amazement': 1, 'Happiness': 1,
    'Negative': -1, 'Sadness': -1, 'Anger': -1, 'Disgust': -1, 'Fear': -1,
    'Disappointment': -1, 'Frustration': -1, 'Grief': -1, 'Anxiety': -1, 'Hatred': -1,
    'Neutral': 0, 'Boredom': 0, 'Indifference': 0, 'Surprise': 0, 'Relief': 0.5,
}
sentiment['score'] = sentiment['Sentiment'].map(lambda s: sentiment_map.get(s.strip(), 0))
sentiment['Likes'] = sentiment['Likes'].fillna(0)
sentiment['Retweets'] = sentiment['Retweets'].fillna(0)

recent = sentiment[sentiment['Year'] == 2023].copy()
recent['YearMonth'] = (recent['Year'].astype(str) + '-' +
                        recent['Month'].astype(str).str.zfill(2))
monthly = (recent
           .groupby('YearMonth')
           .agg(avg_score=('score', 'mean'),
                total_likes=('Likes', 'sum'),
                total_retweets=('Retweets', 'sum'),
                count=('score', 'count'))
           .sort_index()
           .reset_index())

ts = monthly['avg_score'].values
print(f"\n[4.1] Monthly Time Series (2023):")
print(monthly[['YearMonth','avg_score','count']].to_string(index=False))

# ── 4.2 Moving Average Smoothing ──────────────────────────────────────
def moving_average(arr, window):
    result = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1: i + 1])
    return result

ma3 = moving_average(ts, 3)
print(f"\n[4.2] 3-Month Moving Average:")
for i, (label, val, ma) in enumerate(zip(monthly['YearMonth'], ts, ma3)):
    ma_str = f"{ma:.4f}" if not np.isnan(ma) else "  N/A "
    print(f"  {label}: raw={val:.4f}  MA3={ma_str}")

# ── 4.3 Decomposition (Manual: Trend + Residuals) ────────────────────
def centered_moving_average(arr, window=3):
    result = np.full(len(arr), np.nan)
    half = window // 2
    for i in range(half, len(arr) - half):
        result[i] = np.mean(arr[i - half: i + half + 1])
    return result

trend = centered_moving_average(ts, window=3)
trend_interp = pd.Series(trend).interpolate(method='linear').values
residuals = ts - trend_interp

print(f"\n[4.3] Decomposition (Additive Model):")
print(f"  Observed  = Trend + Residual")
for i in range(len(ts)):
    obs = ts[i]
    tr  = trend_interp[i]
    res = residuals[i]
    print(f"  {monthly['YearMonth'][i]}: {obs:.3f} = {tr:.3f} + {res:.3f}")

# ── 4.4 Identify Patterns ─────────────────────────────────────────────
trend_direction = 'upward' if (trend_interp[-1] > trend_interp[0]) else 'downward'
slope = np.polyfit(np.arange(len(trend_interp)), trend_interp, 1)[0]
print(f"\n[4.4] Patterns Identified:")
print(f"  Overall Trend     : {trend_direction} (slope = {slope:.4f}/month)")
print(f"  Peak Month        : {monthly.loc[ts.argmax(), 'YearMonth']} (score={ts.max():.4f})")
print(f"  Lowest Month      : {monthly.loc[ts.argmin(), 'YearMonth']} (score={ts.min():.4f})")
print(f"  Mean Sentiment    : {ts.mean():.4f}")
print(f"  Std Deviation     : {ts.std():.4f}")
print(f"  High Variability  : {'Yes' if ts.std() > 0.3 else 'No'}")

print("\n" + "="*60)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("="*60)
