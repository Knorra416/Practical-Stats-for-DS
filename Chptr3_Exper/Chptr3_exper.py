import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
import statsmodels.api as sm

session_times = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/web_page_data.csv")
session_times.Time = 100 * session_times.Time

ax = session_times.boxplot(by="Page", column="Time")
ax.set_xlabel("")
ax.set_ylabel("Time (in seconds)")
plt.suptitle("")

mean_a = session_times[session_times.Page == "Page A"].Time.mean()
mean_b = session_times[session_times.Page == "Page B"].Time.mean()
print(mean_b - mean_a)


def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[idx_B].mean() - x.loc[idx_A].mean()


perm_diffs = [perm_fun(session_times.Time, 21, 15) for _ in range(1000)]

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x=mean_b - mean_a, color='black', lw=2)
ax.text(50, 190, "Observed\ndifference", bbox={"facecolor": "white"})
ax.set_xlabel("Session time differences (in seconds)")
ax.set_ylabel("Frequency")

obs_pct_diff = 100 * (200 / 23739 - 182 / 22588)
print(f"Observed difference: {obs_pct_diff:.4f}%")
conversion = [0] * 45945
conversion.extend([1] * 382)
conversion = pd.Series(conversion)

perm_diffs = [100 * perm_fun(conversion, 23739, 22588) for _ in range(1000)]

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x=obs_pct_diff, color='black', lw=2)
ax.text(0.06, 200, "Observed\ndifference", bbox={"facecolor": "white"})
ax.set_xlabel("Conversion rate (percent)")
ax.set_ylabel("Frequency")

np.mean([diff > obs_pct_diff for diff in perm_diffs])

survivors = np.array([[200, 23739 - 200], [182, 22588 - 182]])
chi2, p_value, df, _ = stats.chi2_contingency(survivors)
print(f"p-value for single sided test: {p_value/ 2:.4f}")

res = stats.ttest_ind(session_times[session_times.Page == 'Page A'].Time,
                      session_times[session_times.Page == 'Page B'].Time,
                      equal_var=False)
print(f"p-value for single sided test: {res.pvalue / 2:.4f}")

four_sessions = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/four_sessions.csv")
observed_variance = four_sessions.groupby("Page").mean().var()[0]
print(f"Observed Means: {four_sessions.groupby('Page').mean().values.ravel()}")
print(f"Variance: {observed_variance}")


def perm_test(df):
    df = df.copy()
    df["Time"] = np.random.permutation(df["Time"].values)
    return df.groupby("Page").mean().var()[0]


perm_variance = [perm_test(four_sessions) for _ in range(3000)]
print(f"PR(Prob): {np.mean([var > observed_variance for var in perm_variance])}")

click_rate = pd.read_csv("/Users/alexknorr/PycharmProjects/Practical-Stats-for-DS/data/click_rates.csv")
clicks = click_rate.pivot(index='Click', columns='Headline', values='Rate')
box = [1] * 34
box.extend([0] * 2966)
random.shuffle(box)


def chi2(observed, expected):
    pearson_residuals = []
    for row, expect in zip(observed, expected):
        pearson_residuals.append([(observe - expect) ** 2 / expect for observe in row])

    return np.sum(pearson_residuals)


expected_clicks = 34/3
expected_noclicks = 1000 - expected_clicks
expected = [34/3, 1000 - 34 / 3]
chi2observed = chi2(clicks.values, expected)


def perm_fun(box):
    sample_clicks = [sum(random.sample(box, 1000)),
                     sum(random.sample(box, 1000)),
                     sum(random.sample(box, 1000))]
    sample_noclicks = [1000 - n for n in sample_clicks]
    return chi2([sample_clicks, sample_noclicks], expected)


perm_chi2 = [perm_fun(box) for _ in range(2000)]
resampled_p_value = sum(perm_chi2 > chi2observed) / len(perm_chi2)
print(f"Observed chi2: {chi2observed:.4f}")
print(f"Resampled p-value: {resampled_p_value: .4f}")

chisq, pvalue, df, expected = stats.chi2_contingency(clicks)
print(f"Observed chi2: {chisq:.4f}")
print(f"Resampled p-value: {pvalue: .4f}")

effect_size = sm.stats.proportion_effectsize(0.0121, 0.011)
analysis = sm.stats.TTestIndPower()
results = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='larger')
print(f"Sample Size: {results}")
