# edge_case_testing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings("ignore")


# Function to load data
def load_data(url):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url, low_memory=False)
    return df


# Function to display data information
def display_data_info(df):
    print("Data Shape:", df.shape)
    print("Data Types:\n", df.dtypes)
    print("Data Summary:\n", df.describe())
    print("Data Info:\n", df.info())


# Function to handle missing values
def handle_missing_values(df):
    print("Missing Values:\n", df.isnull().sum())
    print("Duplicate Rows:", df.duplicated().sum())


# Function to clean data
def clean_data(df):
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={'education': 'education_Graduate'})
    df['education_Graduate'] = df['education_Graduate'].replace({' Graduate': 1, ' Not Graduate': 0})
    df['loan_status'] = df['loan_status'].replace({' Approved': 1, ' Rejected': 0})
    df['self_employed'] = df['self_employed'].replace({' Yes': 1, ' No': 0})
    return df


# Function for univariate analysis
def univariate_analysis(df):
    columns = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
               'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], edgecolor='black')
        axes[i].set_title(f'Distribution Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        axes[i].hist(df[col], bins=10, edgecolor='black')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.kdeplot(df[col], ax=axes[i], fill=True)
        axes[i].set_title(f'KDE Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
    plt.tight_layout()
    plt.show()


# Function to analyze skewness and kurtosis
def analyze_skewness_kurtosis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("Skewness:\n", df[numeric_cols].skew())
    print("Kurtosis:\n", df[numeric_cols].kurt())


# Function for scaling
def scale_data(df):
    min_max_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    df[['no_of_dependents', 'loan_term', 'cibil_score']] = min_max_scaler.fit_transform(
        df[['no_of_dependents', 'loan_term', 'cibil_score']])
    df[['loan_amount', 'income_annum', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value',
        'bank_asset_value']] = std_scaler.fit_transform(df[['loan_amount', 'income_annum', 'residential_assets_value',
                                                            'commercial_assets_value', 'luxury_assets_value',
                                                            'bank_asset_value']])
    return df


# Function for bivariate analysis
def bivariate_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("Covariance Matrix:\n", df[numeric_cols].cov())
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.title('Correlation Matrix')
    plt.show()

    plt.figure(figsize=(10, 10))
    sns.pairplot(df)
    plt.show()


def main():
    # Load the dataset
    url = 'https://drive.google.com/file/d/1FzPWlJinDMCm4BMn2uY6uiA4fIA1TY4f/view?usp=sharing'
    df = load_data(url)

    # Display data information
    display_data_info(df)

    # Handle missing values and duplicates
    handle_missing_values(df)

    # Clean the data
    df = clean_data(df)

    # Univariate analysis
    univariate_analysis(df)

    # Analyze skewness and kurtosis
    analyze_skewness_kurtosis(df)

    # Scale the data
    df = scale_data(df)

    # Bivariate analysis
    bivariate_analysis(df)


if __name__ == "__main__":
    main()
