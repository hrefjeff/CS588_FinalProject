#!/usr/bin/env python3

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('s1/e1/u2/test-acc.txt', delimiter=';')
unmodified_df = pd.read_csv('s1/e1/u2/test-acc.txt', delimiter=';')

cor_eff = df.corr(numeric_only=True)
plt.figure(figsize = (6,6))
sns.heatmap(cor_eff, linecolor='white', linewidths=1, annot=True)
plt.savefig("corr_heatmap.png", bbox_inches="tight")

plt.figure()
plt.plot(df['acc_x'])
plt.plot(df['acc_y'])
plt.plot(df['acc_z'])

plt.savefig("lineplot.png")
