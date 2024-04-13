#!/usr/bin/env python3

import constants
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():

    exercise = 1
    unit = 5

    # Load dataset for each subject
    for subject in range(1, constants.SUBJECTS + 1):
        df = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test.txt', delimiter=';')

        # Drop the time index as it's not a feature
        df = df.drop(columns=['time index'])

        # Reset the plot
        plt.figure()

        # Plot the values
        plt.plot(df['acc_x'], linestyle='solid', label='X')
        plt.plot(df['acc_y'], linestyle='dotted', label='Y')
        plt.plot(df['acc_z'], linestyle='dashed', label='Z')

        plt.legend()
        filename = f'plots/lineplot/lineplot-s{subject}-e{exercise}-u{unit}.png'
        plt.savefig(filename)

if __name__ == '__main__':
    main()
