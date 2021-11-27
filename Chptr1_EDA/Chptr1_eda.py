import pandas as pd
from scipy.stats import trim_mean

state = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/state.csv")
print(state["Population"].mean())
print(trim_mean(state["Population"], 0.1))
print(state["Population"].median())

import numpy as np
import wquantiles

print(np.average(state["Murder.Rate"], weights=state["Population"]))
print(wquantiles.median(state["Murder.Rate"], weights=state["Population"]))

from statsmodels import robust

print(state["Population"].std())
print(state["Population"].quantile(0.75) - state["Population"].quantile(0.25))
print(robust.scale.mad(state["Population"]))

