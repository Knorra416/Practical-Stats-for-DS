import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

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



