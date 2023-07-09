# Loan Prediction model
- This code loads a loan prediction dataset, performs data preprocessing, transforms numerical attributes, encodes categorical variables, splits the data, and trains a logistic regression model while evaluating its accuracy using cross-validation.

This code performs the following steps:

Imports the required libraries: pandas, numpy, seaborn, and matplotlib.pyplot. It also sets %matplotlib inline to ensure that the plots are displayed inline in the notebook. Additionally, it imports the warnings module and sets a filter to ignore warnings.

Loads the dataset from the "Loan_Prediction" file using pd.read_csv() and assigns it to the variable dt. It then displays the first few rows of the dataset using dt.head().

Fills in missing values in numerical columns (LoanAmount, Loan_Amount_Term, and Credit_History) with their respective means using the fillna() method.

Fills in missing values in categorical columns (Gender, Married, Dependents, and Self_Employed) with their respective modes (most frequent values) using the fillna() method.

Creates a new column Total_Income by summing the ApplicantIncome and CoapplicantIncome columns.

Applies a logarithmic transformation to several attributes (ApplicantIncome, CoapplicantIncome, LoanAmount, and Loan_Amount_Term) to address skewness in their distributions. It stores the transformed values in new columns (ApplicantIncomeLog, CoapplicantIncomeLog, LoanAmountLog, and Loan_Amount_Term_Log).

Plots the distributions of the transformed attributes using sns.distplot().

Computes the correlation matrix between all the numerical attributes of the dataset using dt.corr(). It then visualizes the correlation matrix using sns.heatmap() to show the correlation values as a heatmap.

Drops unnecessary columns (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Total_Income, Loan_ID, and CoapplicantIncomeLog) from the dataset using dt.drop().

Encodes categorical variables (Gender, Married, Education, Self_Employed, Property_Area, Loan_Status, and Dependents) using LabelEncoder() from sklearn.preprocessing.

Splits the dataset into training and testing sets using train_test_split() from sklearn.model_selection.

Defines a function classify() that takes a model, training data, and target variable as input. Inside the function, it performs the model training using model.fit() and calculates the accuracy score using model.score(). It also performs cross-validation using cross_val_score() and displays the accuracy obtained through cross-validation.

Initializes a LogisticRegression model from sklearn.linear_model and calls the classify() function with the logistic regression model, the feature matrix X, and the target variable y.

