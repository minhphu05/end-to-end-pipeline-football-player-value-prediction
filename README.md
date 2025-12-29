# ⚽ Football Player Transfer Value Prediction  
**End-to-End Data Pipeline (2024)**

![](/resource/Transfermarkt_logo.png)

![](/resource/whoscored.jpeg)

## 📌 Overview
This project builds an **end-to-end data science pipeline** to predict **football players’ transfer market value in 2024**.

The pipeline starts from **web crawling real-world football data**, continues with **exploratory data analysis (EDA)** and **data preprocessing**, and ends with **machine learning models** for market value prediction.

Data is collected from:
- **Transfermarkt** – market value, age, position, nationality, club, league
- **WhoScored** – detailed player performance statistics

The goal is to **estimate player transfer value based on performance and personal attributes**, while demonstrating a complete real-world ML workflow.

---

## 🏗️ End-to-End Pipeline

1. **Web Crawling**
2. **Data Cleaning & Integration**
3. **Exploratory Data Analysis (EDA)**
4. **Preprocessing & Feature Engineering**
5. **Machine Learning Modeling**
6. **Model Evaluation**

---

## 🌐 Data Collection (Web Crawling)

### Data Sources
- **Transfermarkt**
  - Market value
  - Age, height
  - Position
  - Nationality
  - Club & league
- **WhoScored**
  - Player ratings
  - Goals, assists
  - Passing, shooting, defensive metrics

### Crawling Tool
- **Selenium**
  - Handles JavaScript-rendered pages
  - Dynamic tables & pagination
  - Anti-bot behavior

### Crawling Workflow
- Automated browser navigation using Selenium WebDriver  
- Crawl player lists by league and season (2023–2024)  
- Extract player-level data  
- Store raw data in CSV format  

---

## 🔍 Exploratory Data Analysis (EDA)

EDA is performed to understand data distribution, correlations, and potential issues.

### EDA Tasks
- Distribution of player market values
- Correlation between performance metrics and market value
- Analysis by player position:
  - Forward
  - Midfielder
  - Defender
  - Goalkeeper
- Missing value inspection
- Outlier detection (extremely high-value players)

### Visualization Tools
- **Matplotlib**
- **Seaborn**

Common visualizations:
- Histograms & KDE plots
- Boxplots by position
- Correlation heatmaps
- Scatter plots between key features and market value

---

## 🧹 Data Preprocessing & Feature Engineering

### Preprocessing
- Handle missing values (imputation or removal)
- Encode categorical features (position, league, nationality)
- Feature scaling (StandardScaler / MinMaxScaler)
- Log-transform target variable (market value) to reduce skewness

### Feature Engineering
- Per-90-minute performance metrics
- Aggregated attacking and defensive indicators
- Age-related nonlinear features

### Libraries
- **Pandas**
- **NumPy**
- **Scikit-learn**

---

## 🤖 Machine Learning Models

Regression models used for market value prediction:

- Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)
- Gradient Boosting / XGBoost (if applicable)

### Training Strategy
- Train / test split
- Cross-validation
- Hyperparameter tuning with:
  - GridSearchCV
  - RandomizedSearchCV

### Evaluation Metrics
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

---

## 📊 Results & Insights

- Player performance metrics strongly correlate with market value
- Age shows a nonlinear effect (peak value at prime age)
- Log transformation significantly improves model performance
- Tree-based models outperform linear models by capturing nonlinear patterns

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Crawling | Selenium |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Language | Python |

---

## 📁 Project Structure

```text
football-value-prediction/
│
├── data/
│   ├── raw/              # Crawled raw data
│   └── processed/        # Cleaned & preprocessed data
│
├── crawling/
│   └── crawl_players.py
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
│
├── models/
│   └── trained_models.pkl
│
├── requirements.txt
└── README.md
