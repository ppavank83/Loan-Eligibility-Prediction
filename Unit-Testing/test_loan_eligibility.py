import unittest
import pandas as pd
from io import StringIO
from loan_eligibility_analysis import import_dataset, clean_data, check_missing_values, check_duplicates, scale_data, compute_skewness, compute_kurtosis, compute_covariance, compute_correlation

class TestLoanEligibility(unittest.TestCase):

    def setUp(self):
        csv_data = """loan_id,no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value,loan_status
                      1,2,Graduate,No,9600000,29900000,12,778,2400000,17600000,22700000,8000000,Approved
                      2,0,Not Graduate,Yes,4100000,12200000,8,417,2700000,2200000,8800000,3300000,Rejected"""
        self.df = pd.read_csv(StringIO(csv_data))
        self.cleaned_df = clean_data(self.df.copy())

    def test_import_dataset(self):
        url = 'https://drive.google.com/file/d/1FzPWlJinDMCm4BMn2uY6uiA4fIA1TY4f/view?usp=sharing'
        df = import_dataset(url)
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        df_cleaned = clean_data(self.df.copy())
        self.assertNotIn('loan_id', df_cleaned.columns)
        self.assertIn('education_Graduate', df_cleaned.columns)

    def test_check_missing_values(self):
        missing_values = check_missing_values(self.df)
        self.assertEqual(missing_values.sum(), 0)

    def test_check_duplicates(self):
        duplicates = check_duplicates(self.df)
        self.assertEqual(duplicates, 0)

    def test_scale_data(self):
        scaled_df = scale_data(self.cleaned_df.copy())
        self.assertAlmostEqual(scaled_df['no_of_dependents'].max(), 1.0)
        self.assertAlmostEqual(scaled_df['no_of_dependents'].min(), 0.0)

    def test_compute_skewness(self):
        skewness = compute_skewness(self.cleaned_df)
        self.assertIsInstance(skewness, pd.Series)

    def test_compute_kurtosis(self):
        kurtosis = compute_kurtosis(self.cleaned_df)
        self.assertIsInstance(kurtosis, pd.Series)

    def test_compute_covariance(self):
        covariance = compute_covariance(self.cleaned_df)
        self.assertIsInstance(covariance, pd.DataFrame)

    def test_compute_correlation(self):
        correlation = compute_correlation(self.cleaned_df)
        self.assertIsInstance(correlation, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
