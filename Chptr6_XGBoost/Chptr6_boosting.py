import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

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





