import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import itertools

loan3000 = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/loan3000.csv")

predictors = ["borrower_score", "payment_inc_ratio"]
outcome = "outcome"

X = loan3000[predictors]
y = pd.Series([1 if o == 'default' else 0 for o in loan3000[outcome]])

xgb = XGBClassifier(objective="binary:logistic", subsample=0.63, use_label_encoder=False, eval_metric='error')
xgb.fit(X, y)

xgb_df = X.copy()
xgb_df['prediction'] = ['default' if p == 1 else 'paid off' for p in xgb.predict(X)]
xgb_df['prob_default'] = xgb.predict_proba(X)[:, 0]
print(xgb_df.head())

fig, ax = plt.subplots(figsize=(6, 4))
xgb_df.loc[xgb_df.prediction == "paid off"].plot(
    x="borrower_score", y="payment_inc_ratio", style=".", markerfacecolor="none", markeredgecolor="C1", ax=ax
)
xgb_df.loc[xgb_df.prediction == "default"].plot(
    x="borrower_score", y="payment_inc_ratio", style="o", markerfacecolor="none", markeredgecolor="C0", ax=ax
)
ax.legend(["paid off", "default"])
ax.set_xlim(0, 1)
ax.set_ylim(0, 25)
ax.set_xlabel("borrower_score")
ax.set_ylabel("payment_inc_ratio")

loan_data = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/loan_data.csv.gz")
loan_data = loan_data.drop(columns=['Unnamed: 0', 'status'])
loan_data['outcome'] = pd.Categorical(loan_data['outcome'],
                                      categories=['paid off', 'default'],
                                      ordered=True)
predictors = ["loan_amnt", "term", "annual_inc", "dti", "payment_inc_ratio", "revol_bal", "revol_util", "purpose",
              "delinq_2yrs_zero", "pub_rec_zero", "open_acc", "grade", "emp_length", "purpose_", "home_", "emp_len_",
              "borrower_score"]
outcome = "outcome"
X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = pd.Series([1 if o == "default" else 0 for o in loan_data[outcome]])
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=10000)
xgb_default = XGBClassifier(objective="binary:logistic", n_estimators=250, max_depth=6, reg_lambda=0, learning_rate=0.3,
                            subsample=1)
xgb_default.fit(train_X, train_y)
pred_default = xgb_default.predict_proba(valid_X)[:, 1]
error_default = abs(valid_y - pred_default) > 0.5
print(f"default: {np.mean(error_default)}")

xgb_penalty = XGBClassifier(objective="binary:logistic", n_estimators=250, max_depth=6, reg_lambda=1000,
                            learning_rate=0.1, subsample=0.63)
xgb_penalty.fit(train_X, train_y)
pred_penalty = xgb_penalty.predict_proba(valid_X)[:, 1]
error_penalty = abs(valid_y - pred_penalty) > 0.5
print(f"penalty: {np.mean(error_penalty)}")

results = []
for i in range(1, 250):
    train_default = xgb_default.predict_proba(train_X, ntree_limit=i)[:, 1]
    train_penalty = xgb_penalty.predict_proba(train_X, ntree_limit=i)[:, 1]
    pred_default = xgb_default.predict_proba(valid_X, ntree_limit=i)[:, 1]
    pred_penalty = xgb_penalty.predict_proba(valid_X, ntree_limit=i)[:, 1]
    results.append({
        "iterations": i,
        "default train": np.mean(abs(train_y - train_default) > 0.5),
        "penalty train": np.mean(abs(train_y - train_penalty) > 0.5),
        "default test": np.mean(abs(valid_y - pred_default) > 0.5),
        "penalty test": np.mean(abs(valid_y - pred_penalty) > 0.5),
    })
results = pd.DataFrame(results)
print(results.head())

ax = results.plot(x="iterations", y="default test")
results.plot(x="iterations", y="penalty test", ax=ax)
results.plot(x="iterations", y="default train", ax=ax)
results.plot(x="iterations", y="penalty train", ax=ax)

idx = np.random.choice(range(5), size=len(X), replace=True)
error = []
for eta, max_depth in itertools.product([0.1, 0.5, 0.9], [3, 6, 9]):
    xgb = XGBClassifier(objective="binary:logistic", n_estimators=250, max_depth=max_depth, learning_rate=eta)
    cv_error = []
    for k in range(5):
        fold_idx = idx == k
        train_X = X.loc[~fold_idx]
        train_y = y[~fold_idx]
        valid_X = X.loc[fold_idx]
        valid_y = y[fold_idx]

        xgb.fit(train_X, train_y)
        pred = xgb.predict_proba(valid_X)[:, 1]
        cv_error.append(np.mean(abs(valid_y - pred) > 0.5))
    error.append({
        "eta": eta,
        "max_depth": max_depth,
        "avg_error": np.mean(cv_error)
    })
    print(error[-1])
errors = pd.DataFrame(error)
print(errors)
print(errors.pivot_table(index="eta", columns="max_depth", values="avg_error") * 100)


