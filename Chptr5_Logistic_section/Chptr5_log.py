import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

loan_data = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/loan_data.csv.gz")

loan_data.outcome = loan_data.outcome.astype('category')
loan_data.outcome.cat.reorder_categories(['paid off', 'default'])
loan_data.purpose_ = loan_data.purpose_.astype('category')
loan_data.home_ = loan_data.home_.astype('category')
loan_data.emp_len_ = loan_data.emp_len_.astype('category')

predictors = ["payment_inc_ratio", "purpose_", "home_", "emp_len_", "borrower_score"]
outcome = "outcome"

X = pd.get_dummies(loan_data[predictors], prefix="", prefix_sep="", drop_first=True)
y = loan_data[outcome]

logit_reg = LogisticRegression(penalty="l2", C=1e42, solver="liblinear")
logit_reg.fit(X, y)

logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)

pred = pd.DataFrame(logit_reg.predict_log_proba(X), columns=loan_data[outcome].cat.categories)
print(pred.describe())

pred_prob = pd.DataFrame(logit_reg.predict_proba(X), columns=loan_data[outcome].cat.categories)
print(pred_prob.describe())

y_numbers = [1 if yi == "default" else 0 for yi in y]
logit_reg_sm = sm.GLM(y_numbers, X.assign(const=1), family=sm.families.Binomial())
logit_result = logit_reg_sm.fit()
print(logit_result.summary())

