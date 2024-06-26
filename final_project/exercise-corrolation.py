#!/usr/bin/env python3

import constants
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():

    exercise = 1
    unit = 2

    # Load dataset for each subject
    for subject in range(1, constants.SUBJECTS + 1):
        df = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test.txt', delimiter=';')

        # Drop the time index as it's not a feature
        df = df.drop(columns=['time index'])

        # Extract the corrolation coefficient
        cor_eff = df.corr(numeric_only=True)
        # print('\n')
        # print(f'Corrolation matrix for subject {subject}')
        # print(cor_eff)

        # Plot the corrolation heatmap
        plt.figure(figsize = (8,8))
        sns.heatmap(cor_eff, linecolor='white', linewidths=1, annot=True)
        plt.savefig(f'plots/corr_heatmap-s{subject}-e{exercise}-u{unit}.png')

if __name__ == '__main__':
    main()
