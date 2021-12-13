import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from dmba import stepwise_selection
from dmba import AIC_score

lung = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/LungDisease.csv")
predictors = ["Exposure"]
outcome = 'PEFR'

model = LinearRegression()
model.fit(lung[predictors], lung[outcome])
print(f"Intercept: {model.intercept_:.3f}")
print(f"Coefficient Exposure: {model.coef_[0]:.3f}")

fitted = model.predict(lung[predictors])
residuals = lung[outcome] - fitted
print(f"Fitted values: {fitted}")
print(f"Residuals: {residuals}")

subset = ['AdjSalePrice', 'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
house = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/house_sales.csv", sep='\t')
print(house[subset].head())
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = "AdjSalePrice"

house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])
print(f"Intercept: {house_lm.intercept_}")
print(f"Coefficients:")
for name, coef in zip(predictors, house_lm.coef_):
    print(f"{name}: {coef}")

fitted = house_lm.predict(house[predictors])
RMSE = np.sqrt(mean_squared_error(house[outcome], fitted))
r2 = r2_score(house[outcome], fitted)
print(f"RMSE: {RMSE}")
print(f"r2: {r2}")
# Statsmodels has more detailed regression summaries


predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade', 'PropertyType', 'NbrLivingUnits',
              'SqFtFinBasement', 'YrBuilt', 'YrRenovated', 'NewConstruction']
X = pd.get_dummies(house[predictors], drop_first=True)
X["NewConstruction"] = [1 if nc else 0 for nc in X["NewConstruction"]]

house_full = sm.OLS(house[outcome], X.assign(const=1))
results = house_full.fit()
print(results.summary())

y = house[outcome]


def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(X[variables], y)
    return model


def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(y, [y.mean()] * len(y), model, df=1)
    return AIC_score(y, model.predict(X[variables]), model)


best_model, best_variables = stepwise_selection(X.columns, train_model, score_model, verbose=True)
print(f"Intercept: {best_model.intercept_}")
print(f"Coefficients:")
for name, coef in zip(best_variables, best_model.coef_):
    print(f"{name}: {coef}")

house["Year"] = [int(date.split("-")[0]) for date in house.DocumentDate]
house["weight"] = house.Year - 2005
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = "AdjSalePrice"

house_wt = LinearRegression()
house_wt.fit(house[predictors], house[outcome], sample_weight=house.weight)
print(house_wt.coef_)
print(house_wt.intercept_)

print(pd.get_dummies(house["PropertyType"]).head())
print(pd.get_dummies(house["PropertyType"], drop_first=True).head())

print(pd.DataFrame(house["ZipCode"].value_counts()).transpose())

zip_groups = pd.DataFrame([
    *pd.DataFrame({
        "ZipCode": house["ZipCode"],
        "residual": house[outcome] - house_lm.predict(house[predictors])
    })
    .groupby(["ZipCode"])
    .apply(lambda x: {
        "ZipCode": x.iloc[0, 0],
        "count": len(x),
        "median_residual": x.residual.median()
    })
]).sort_values("median_residual")
zip_groups["cum_count"] = np.cumsum(zip_groups["count"])
zip_groups["ZipGroup"] = pd.qcut(zip_groups["cum_count"], 5, labels=False, retbins=False)
to_join = zip_groups[["ZipCode", "ZipGroup"]].set_index("ZipCode")
house = house.join(to_join, on="ZipCode")
house["ZipGroup"] = house["ZipGroup"].astype("category")

