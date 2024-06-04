# Loan-Eligibility-Prediction

Note: This repository was created as part of our semester project in the Ai and Data Science program (PGWD) at Loyalist College in Toronto.

# Problem Statement: 
In the modern financial landscape, the influx of loan applications poses a significant challenge for institutions. Manual evaluation of these applications is time-consuming and prone to errors. To streamline this process, our task is to leverage machine learning techniques and algorithms to build a loan eligibility prediction system to help aid and streamline the process of approve or reject loan request.

#Technologies Used:
1) Co-Lab
2) Python
   libraries: Pandas,
              Numpy,
              Matplotlib,
              Seaborn.

# The Flow of project is As Follows:

1.Problem Definition and Data Collection:
  	  a) Clearly define the problem we want to solve.
	  b) Collect the Dataset
2.Data Preprocessing:
	  a) Clean the data.
3.Exploratory Data Analysis (EDA):
	  a) Visualize the data (histograms, scatter plots, etc.).
	  b) Understand the distribution of features.
	  c) Identify patterns, correlations, and anomalies.
4.Feature Engineering:
	  a) Select relevant features based on domain knowledge and Insights from EDA.
5.Data Splitting:
	  a) Divide the data into training, validation, and test sets.
6.Model Selection:
	  a) Choose appropriate algorithms
7.Model Training & Evaluation:
	  a) Train the selected model using the training data.
	  b) Evaluate the model’s performance using validation data (accuracy, precision, recall, etc.).
8.Model Tuning:
	  a) Optimize model hyperparameters 
9.Model Validation and Visualization:
	  a) Validate the tuned model on the test set.
	  b) Visualize model predictions and compare them with actual values.
10.Conclusions:
	  a) Analyse the model’s performance.
	  b) Draw conclusions based on the results.


# Data Set:
The dataset we're using here is from Kaggel (i.e. https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

Features of the Dataset
      FEATURE NAME                	  DESCRIPTION
1.  loan_id			        :Unique Id of Loan
2.  no_of_dependents 	          	:Number of Dependents of the Applicant
3.  education		                :Education of the Applicant (Graduate/Not Graduate)
4.  self_employed	              	:Employment Status of the Applicant (Yes/No)
5.  income_annum	              	:Annual Income of the Applicant
6.  loan_amount		              	:Applied Loan Amount
7.  loan_term			        :Loan Term in Years
8.  cibil_score		              	:Credit Score of the applicant
9.  residential_assets_value	        :Residential Assets Value of the Applicant
10. commercial_assets_value	        :Commerical Assets Value of the Applicant	
11. luxury_assets_value		        :Luxury Assets Value of the Applicant
12. bank_asset_value		        :Current Bank Balance of the Applicant
13. loan_status			        :Loan Approval Status (Approved/Rejected)

# Current Status

We have completed bivariate analysis and are now working on multivariate analysis.

# Work completed until now 

1. Our data has 4269 rows and 13 unique features.
2. Our data contains no null values.
3. changed our object features (e.g., education, self_employed, loan_status) to "1" or "0" (fortunately, all object type features have just two unique values).
4. Univariant Analysis: Visualized the distribution of Features using Histogram, kernel density estimate (KDE) plot, Box plots,etc..
5. Bivariate Analysis: Measured and plotted Correlation Matrix to understand relationship between two features.

# Observations (*As of now)

->Income and Loan Dynamics:
	Strong linear relationship between income and loan amount. Higher income individuals are more likely to obtain larger loans and possess valuable assets.
->CIBIL Score and Loan Status:
  CIBIL score plays a critical role in determining loan status, as evidenced by their strong positive correlation and covariance. A good CIBIL score increases the likelihood of a favorable loan status.
->Asset Values:
	Luxury and residential asset values are strongly associated with both income and loan amount. Applicants with higher incomes and larger loans tend to report higher asset values, indicating financial stability and creditworthiness.
  
->Visual Patterns:
	Pair plot confirms the linear relationships observed in the correlation matrix and highlights the distribution of data points. Clear trends in scatter plots align with statistical measures, providing a holistic understanding of data relationships.



