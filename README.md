# Loan-prediction-using-Machine-Learning-and-Python

## Aim

Our aim from the project is to make use of pandas, matplotlib, & seaborn libraries from python to extract insights from the data and xgboost, & scikit-learn libraries for machine learning.

Secondly, to learn how to hypertune the parameters using grid search cross validation for the xgboost machine learning model.

And in the end, to predict whether the loan applicant can replay the loan or not using voting ensembling techniques of combining the predictions from multiple machine learning algorithms.

## Attributes in the dataset
Loan_ID - Unique Loan ID

**Gender** - Male/ Female

**Married**- Applicant married (Y/N)

**Dependents** - Number of dependents

**Education** - Applicant Education (Graduate/ Under Graduate)

**Self_Employed** - Self employed (Y/N)

**ApplicantIncome **- Applicant income

**CoapplicantIncome** - Coapplicant income

**LoanAmount** - Loan amount in thousands

**Loan_Amount_Term** - Term of loan in months

**Credit_History** - credit history meets guidelines

**Property_Area**- Urban/ Semi Urban/ Rural

**Loan_Status** - Loan approved (Y/N)

## Major observation from the data

Applicants who are male and married tends to have more applicant income whereas applicant who are female and married have least applicant income

Applicants who are male and are graduated have more applicant income over the applicants who have not graduated.

Again the applicants who are married and graduated have the more applicant income.

Applicants who are not self employed have more applicant income than the applicants who are self employed.

Applicants who have more dependents have least applicant income whereas applicants which have no dependents have maximum applicant income.

Applicants who have property in urban and have credit history have maximum applicant income

Applicants who are graduate and have credit history have more applicant income.

Loan Amount is linearly dependent on Applicant income

From heatmaps, applicant income and loan amount are highly positively correlated.

Male applicants are more than female applicants.

No of applicants who are married are more than no of applicants who are not married.

Applicants with no dependents are maximum.

Applicants with graduation are more than applicants whith no graduation.

Property area is to be find more in semi urban areas and minimum in rural areas.

## Feature Engineering
### Converting the scale of loan term from months to years
df['Loan_Amount_Term']=df['Loan_Amount_Term']/12
### Adding the applicant and co-applicant income to get the total income per application
df['total_income']=df['ApplicantIncome'] + df['CoapplicantIncome']

## Importance Feature

	Importance	Feature 
 
7	0.837136	Credit_History

8	0.127906	Loan_Status

10	0.034958	Property_Area_Rural

0	0.000000	Gender

1	0.000000	Married

2	0.000000	Dependents

3	0.000000	Education

4	0.000000	Self_Employed

5	0.000000	LoanAmount

6	0.000000	Loan_Amount_Term

9	0.000000	total_income

11	0.000000	Property_Area_Semiurban


# Model Accuarcy

**Random Forest** : accuracy_score = 0.7680412371134021

**Logistic Regression**: accuracy_score = 0.7783505154639175

**Decision Trees**: accuracy_score = 0.788659793814433

**K Nearest Neighbor Classifier**: accuracy_score = 0.5979381443298969

**Extreme Gradient Boosting Classifier**: accuracy_score = 0.7783505154639175
