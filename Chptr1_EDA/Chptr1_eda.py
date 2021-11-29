import pandas as pd
from scipy.stats import trim_mean
import numpy as np
import wquantiles
from statsmodels import robust
import matplotlib.pyplot as plt
import seaborn as sns

state = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/state.csv")

print(state["Population"].mean())
print(trim_mean(state["Population"], 0.1))
print(state["Population"].median())

print(np.average(state["Murder.Rate"], weights=state["Population"]))
print(wquantiles.median(state["Murder.Rate"], weights=state["Population"]))

print(state["Population"].std())
print(state["Population"].quantile(0.75) - state["Population"].quantile(0.25))
print(robust.scale.mad(state["Population"]))

print(state["Murder.Rate"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]))
ax = (state["Population"]/1_000_000).plot.box()
ax.set_ylabel("Population (millions)")

binnedPopulation = pd.cut(state["Population"], 10)
binnedPopulation.value_counts()

ax = (state["Population"]/1_000_000).plot.hist(figsize=(4, 4))
ax.set_xlabel("Population (millions)")

ax = state["Murder.Rate"].plot.hist(density=True, xlim=[0, 12], bins=range(1,12))
state["Murder.Rate"].plot.density(ax=ax)
ax.set_xlabel("Murder Rate (per 100_000)")

dfw = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/dfw_airline.csv")

ax = dfw.transpose().plot.bar(figsize=(4, 4), legend=False)
ax.set_xlabel("Cause of Delay")
ax.set_ylabel("Count")

sp500_px = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/sp500_data.csv.gz", index_col=0)
sp500_sym = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/sp500_sectors.csv")

etfs = sp500_px.loc[sp500_px.index > "2021-07-01", sp500_sym[sp500_sym["sector"] == 'etf']["symbol"]]
sns.heatmap(etfs.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True))

telecomSymbols = sp500_sym[sp500_sym['sector'] == 'telecommunications_services']['symbol']
# Filter data for dates July 2012 through June 2015
telecom = sp500_px.loc[sp500_px.index >= '2012-07-01', telecomSymbols]
ax = telecom.plot.scatter(x="T", y="VZ", figsize=(4, 4), marker="$\u25EF$")
ax.set_xlabel("ATT (T)")
ax.set_ylabel("Verizon (VZ)")
ax.axhline(0, color="grey", lw=1)
ax.axvline(0, color="grey", lw=1)

kc_tax = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/kc_tax.csv.gz")
kc_tax0 = kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) &
                     (kc_tax.SqFtTotLiving > 100) &
                     (kc_tax.SqFtTotLiving < 3500), :]
print(kc_tax0.shape)

ax = kc_tax0.plot.hexbin(x="SqFtTotLiving", y="TaxAssessedValue", gridsize=30, sharex=False, figsize=(5, 4))
ax.set_xlabel("Finished Square Feet")
ax.set_ylabel("Tax-Assessed Value")

lc_loans = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/lc_loans.csv")
crosstab = lc_loans.pivot_table(index='grade', columns='status', aggfunc=lambda x: len(x), margins=True)
df = crosstab.loc['A':'G', :].copy()
df.loc[:, "Charged Off": 'Late'] = df.loc[:, "Charged Off": "Late"].div(df["All"], axis=0)
df["All"] = df["All"]/sum(df["All"])
perc_crosstab = df
print(perc_crosstab)

airline_stats = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/airline_stats.csv")
ax = airline_stats.boxplot(by='airline', column='pct_carrier_delay')


zip_codes = [98188, 98105, 98108, 98126]
kc_tax_zip = kc_tax0.loc[kc_tax0.ZipCode.isin(zip_codes),:]


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=25, cmap=cmap, **kwargs)


g = sns.FacetGrid(kc_tax_zip, col='ZipCode', col_wrap=2)
g.map(hexbin, 'SqFtTotLiving', 'TaxAssessedValue', extent=[0, 3500, 0, 700000])
g.set_axis_labels('Finished Square Feet', 'Tax-Assessed Value')
g.set_titles("Zip Code {col_name: .0f}")





