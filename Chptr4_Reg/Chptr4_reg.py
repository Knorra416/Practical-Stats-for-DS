import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from dmba import stepwise_selection
from dmba import AIC_score
from pygam import LinearGAM, s, l

matplotlib.use('MacOSX')
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

model = smf.ols(formula="AdjSalePrice ~ SqFtTotLiving*ZipGroup + SqFtLot + Bathrooms "
                        "+ Bedrooms + BldgGrade + PropertyType", data=house)
results = model.fit()
print(results.summary())

house_98105 = house.loc[house["ZipCode"] == 98105, ]
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = "AdjSalePrice"

house_outlier = sm.OLS(house_98105[outcome], house_98105[predictors].assign(const=1))
result_98105 = house_outlier.fit()
influence = OLSInfluence(result_98105)
sresiduals = influence.resid_studentized_internal
sresiduals.idxmin(), sresiduals.min()
outlier = house_98105.loc[sresiduals.idxmin(), :]
print(f"Adjsaleprice: {outlier[outcome]}")
print(outlier[predictors])

influence = OLSInfluence(result_98105)
fig, ax = plt.subplots(figsize=(5, 5))
ax.axhline(-2.5, linestyle="--", color="C1")
ax.axhline(2.5, linestyle="--", color="C1")
ax.scatter(influence.hat_matrix_diag, influence.resid_studentized_internal,
           s=1000 * np.sqrt(influence.cooks_distance[0]), alpha=0.5)
ax.set_xlabel("Hat Values")
ax.set_ylabel("Studentized Residuals")
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
sns.regplot(result_98105.fittedvalues, np.abs(result_98105.resid), scatter_kws={"alpha": 0.25}, line_kws={"color": "C1"},
            lowess=True, ax=ax)

sm.graphics.plot_ccpr(result_98105, 'SqFtTotLiving')

model_poly = smf.ols(formula="AdjSalePrice ~ SqFtTotLiving + I(SqFtTotLiving**2) + SqFtLot + Bathrooms + Bedrooms +"
                             "BldgGrade", data=house_98105)
results_poly = model_poly.fit()
print(results_poly.summary())

formula = 'AdjSalePrice ~ bs(SqFtTotLiving, df=6, degree=3) + SqFtLot + Bathrooms + Bedrooms + BldgGrade'
model_spline = smf.ols(formula=formula, data=house_98105)
result_spline = model_spline.fit()
print(result_spline.summary())

predictors = ["SqFtTotLiving", "SqFtLot", "Bathrooms", "Bedrooms", "BldgGrade"]
outcome = "AdjSalePrice"
X = house_98105[predictors].values
Y = house_98105[outcome]

gam = LinearGAM(s(0, n_splines=12) + l(1) + l(2) + l(3) + l(4))
gam.gridsearch(X, Y)
print(gam.summary())
