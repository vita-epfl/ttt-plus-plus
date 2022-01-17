"""System module."""
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd

ttt_toy = pd.read_csv("docs/result.csv")
# discrete
for i in range(ttt_toy.shape[0]):
    if ttt_toy.iloc[i, 5] < 0.5:
        ttt_toy.iloc[i, 5] = 0.4
    if 0.5 <= ttt_toy.iloc[i, 5] <= 0.6:
        ttt_toy.iloc[i, 5] = 0.55
    if 0.6 < ttt_toy.iloc[i, 5] <= 0.7:
        ttt_toy.iloc[i, 5] = 0.65
    if 0.7 < ttt_toy.iloc[i, 5] <= 0.8:
        ttt_toy.iloc[i, 5] = 0.75
    if 0.8 < ttt_toy.iloc[i, 5] <= 0.9:
        ttt_toy.iloc[i, 5] = 0.85
    if 0.9 < ttt_toy.iloc[i, 5] <= 1.0:
        ttt_toy.iloc[i, 5] = 0.95

params = {'legend.fontsize': 12,
          'figure.figsize': (15, 5),
          'axes.labelsize': 14,
          'axes.titlesize': 14,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11}
pylab.rcParams.update(params)


# Test Accuracy (Domain Shift) ana Adapt Accuracy
fig, ax = plt.subplots()
fig.set_size_inches(3, 3.8)
sns.lineplot(data=ttt_toy, x="test", y="TTT", ci="sd")
sns.lineplot(data=ttt_toy, x="test", y="TTT++", ci="sd")
sns.lineplot(data=ttt_toy, x="test", y="TENT", ci="sd")
sns.lineplot(data=ttt_toy, x="test", y="SHOT", ci="sd")
plt.legend(labels=['TTT', 'TTT++', 'TENT', 'SHOT'], loc=4)
plt.xlim([0.4, 0.95])
plt.ylim([0.4, 1.0])
plt.xlabel('Test Accuracy')
plt.ylabel('Adapt Accuracy')
plt.grid()
fig.savefig('docs/moon_shift.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Task relation and Adapt Accuracy
fig, ax = plt.subplots()
fig.set_size_inches(3, 3.8)
sns.lineplot(data=ttt_toy, x='Relation', y='TTT')
sns.lineplot(data=ttt_toy, x='Relation', y='TTT++')
plt.xlim([0.744, 0.896])
plt.ylim([0.6, 0.9])
plt.legend(labels=['TTT', 'TTT++'], loc=2)
plt.xlabel('Task Relation')
plt.ylabel('Adapt Accuracy')
plt.grid()
fig.savefig('docs/moon_relation.png', bbox_inches='tight', pad_inches=0)
plt.close(fig)
