import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Function to import the dataset
def import_dataset(url):
    file_id = url.split('/')[-2]
    dwn_url = 'https://drive.google.com/uc?id=' + file_id
    df = pd.read_csv(dwn_url, low_memory=False)
    return df

# Function to clean the data
def clean_data(df):
    df = df.drop(['loan_id'], axis=1)
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={'education': 'education_Graduate'})
    df['education_Graduate'] = df['education_Graduate'].replace({' Graduate': 1, ' Not Graduate': 0})
    df['loan_status'] = df['loan_status'].replace({' Approved': 1, ' Rejected': 0})
    df['self_employed'] = df['self_employed'].replace({' Yes': 1, ' No': 0})
    return df

# Function to check for missing values
def check_missing_values(df):
    return df.isnull().sum()

# Function to check for duplicate rows
def check_duplicates(df):
    return df.duplicated().sum()

# Function to scale the data
def scale_data(df):
    min_max_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    df[['no_of_dependents', 'loan_term', 'cibil_score']] = min_max_scaler.fit_transform(df[['no_of_dependents', 'loan_term', 'cibil_score']])
    df[['loan_amount', 'income_annum', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']] = (
        std_scaler.fit_transform(df[['loan_amount', 'income_annum', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']]))
    return df

# Function to plot distributions
def plot_distributions(df, columns):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], edgecolor='black')
        axes[i].set_title(f'Distribution Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Function to plot histograms
def plot_histograms(df, columns):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        axes[i].hist(df[col], bins=10, edgecolor='black')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Function to plot box plots
def plot_box_plots(df, columns):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('')
    plt.tight_layout()
    plt.show()

# Function to plot KDE plots
def plot_kde_plots(df, columns):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.kdeplot(df[col], ax=axes[i], fill=True)
        axes[i].set_title(f'KDE Plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
    plt.tight_layout()
    plt.show()

# Function to compute skewness
def compute_skewness(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].skew()

# Function to compute kurtosis
def compute_kurtosis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].kurt()

# Function to compute covariance
def compute_covariance(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].cov()

# Function to compute correlation
def compute_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()

# Function to plot correlation matrix
def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True)
    plt.title('Correlation Matrix')
    plt.show()

# Function to plot pairplot
def plot_pairplot(df):
    plt.figure(figsize=(10, 10))
    sns.pairplot(df)
    plt.show()
