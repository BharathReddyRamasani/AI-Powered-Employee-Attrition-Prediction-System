"""
Data loader module for IBM HR Analytics Employee Attrition dataset.
Handles data loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import os

# ── Column definitions ────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

ORDINAL_COLS_DROP = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']

NUMERIC_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the raw IBM HR Analytics dataset."""
    # Try multiple paths
    paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv'),
        'data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
        'WA_Fn-UseC_-HR-Employee-Attrition.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)

    # Fallback: generate synthetic data matching IBM HR dataset structure
    return _generate_synthetic_data()


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic IBM HR-like dataset for demo purposes."""
    np.random.seed(42)
    n = 1470

    departments = ['Sales', 'Research & Development', 'Human Resources']
    dept_weights = [0.304, 0.653, 0.043]

    job_roles = {
        'Sales': ['Sales Executive', 'Sales Representative', 'Manager'],
        'Research & Development': ['Research Scientist', 'Laboratory Technician',
                                   'Manufacturing Director', 'Research Director',
                                   'Manager', 'Healthcare Representative'],
        'Human Resources': ['Human Resources', 'Manager']
    }

    education_fields = ['Life Sciences', 'Medical', 'Marketing',
                        'Technical Degree', 'Human Resources', 'Other']

    dept_col = np.random.choice(departments, n, p=dept_weights)
    job_role_col = [np.random.choice(job_roles[d]) for d in dept_col]

    age = np.random.randint(18, 61, n)
    monthly_income = np.random.randint(1009, 19999, n)
    job_satisfaction = np.random.randint(1, 5, n)
    years_at_company = np.random.randint(0, 41, n)
    overtime = np.random.choice(['Yes', 'No'], n, p=[0.284, 0.716])

    # Attrition logic: higher risk if low satisfaction, low income, overtime
    attrition_prob = (
        0.05 +
        (job_satisfaction == 1) * 0.15 +
        (monthly_income < 3000) * 0.10 +
        (overtime == 'Yes') * 0.12 +
        (years_at_company < 2) * 0.08
    )
    attrition_prob = np.clip(attrition_prob, 0, 1)
    attrition = np.where(np.random.random(n) < attrition_prob, 'Yes', 'No')

    df = pd.DataFrame({
        'Age': age,
        'Attrition': attrition,
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
                                           n, p=[0.709, 0.188, 0.103]),
        'DailyRate': np.random.randint(102, 1499, n),
        'Department': dept_col,
        'DistanceFromHome': np.random.randint(1, 30, n),
        'Education': np.random.randint(1, 6, n),
        'EducationField': np.random.choice(education_fields, n),
        'EmployeeCount': 1,
        'EmployeeNumber': np.arange(1, n + 1),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n),
        'Gender': np.random.choice(['Male', 'Female'], n, p=[0.600, 0.400]),
        'HourlyRate': np.random.randint(30, 100, n),
        'JobInvolvement': np.random.randint(1, 5, n),
        'JobLevel': np.random.randint(1, 6, n),
        'JobRole': job_role_col,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'],
                                          n, p=[0.319, 0.459, 0.222]),
        'MonthlyIncome': monthly_income,
        'MonthlyRate': np.random.randint(2094, 26999, n),
        'NumCompaniesWorked': np.random.randint(0, 10, n),
        'Over18': 'Y',
        'OverTime': overtime,
        'PercentSalaryHike': np.random.randint(11, 26, n),
        'PerformanceRating': np.random.choice([3, 4], n, p=[0.839, 0.161]),
        'RelationshipSatisfaction': np.random.randint(1, 5, n),
        'StandardHours': 80,
        'StockOptionLevel': np.random.randint(0, 4, n),
        'TotalWorkingYears': np.random.randint(0, 41, n),
        'TrainingTimesLastYear': np.random.randint(0, 7, n),
        'WorkLifeBalance': np.random.randint(1, 5, n),
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': np.random.randint(0, 19, n),
        'YearsSinceLastPromotion': np.random.randint(0, 16, n),
        'YearsWithCurrManager': np.random.randint(0, 18, n),
    })
    return df


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the raw dataframe for ML.
    Returns: (X_train, X_test, y_train, y_test, feature_names, df_encoded)
    """
    df = df.copy()

    # Drop constant / identifier columns
    df.drop(columns=[c for c in ORDINAL_COLS_DROP if c in df.columns], inplace=True)

    # Encode target
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

    # Label encode categoricals
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [c for c in df.columns if c != 'Attrition']
    X = df[feature_cols].values
    y = df['Attrition'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_cols, scaler, df


@st.cache_data(show_spinner=False)
def get_eda_stats(df: pd.DataFrame) -> dict:
    """Compute key EDA statistics from raw dataframe."""
    total = len(df)
    attrition_yes = (df['Attrition'] == 'Yes').sum()
    attrition_no = (df['Attrition'] == 'No').sum()
    attrition_rate = attrition_yes / total * 100

    dept_attrition = (
        df.groupby('Department')['Attrition']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'Attrition': 'Attrition_Rate'})
    )

    gender_attrition = (
        df.groupby('Gender')['Attrition']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'Attrition': 'Attrition_Rate'})
    )

    income_yes = df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean()
    income_no = df[df['Attrition'] == 'No']['MonthlyIncome'].mean()

    return {
        'total': total,
        'attrition_yes': int(attrition_yes),
        'attrition_no': int(attrition_no),
        'attrition_rate': round(attrition_rate, 2),
        'dept_attrition': dept_attrition,
        'gender_attrition': gender_attrition,
        'income_yes': round(income_yes, 0),
        'income_no': round(income_no, 0),
        'avg_age': round(df['Age'].mean(), 1),
        'avg_tenure': round(df['YearsAtCompany'].mean(), 1),
    }
