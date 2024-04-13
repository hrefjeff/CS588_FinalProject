#!/usr/bin/env python3

import constants
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():

    exercise = 1
    unit = 2

    # Load dataset for each subject
    for subject in range(1, constants.SUBJECTS + 1):
        df = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test.txt', delimiter=';')

        # Drop the time index as it's not a feature
        df = df.drop(columns=['time index'])

        # Preprocessing - Standardize the data
        scaler = StandardScaler()
        normalized_df = scaler.fit_transform(df)

        # Finding the principle components
        pca = PCA(n_components=5)
        principalComponents = pca.fit_transform(normalized_df)

        # Display contribution of each pc's
        ev = pca.explained_variance_ratio_
        # print(f'{subject} : {ev}')

        # These values are normalized to a scale of 1.
        # The first principal component retains .61 (61%) of information
        # The second principal component retains .22 (22%) of information
        # The third principal component retains .11 (11%) of information
        # The fourth principal component retains .04 (4%) of information
        # The fifth principal component retains .003 (.3%) of information
        # So with the first 5 PC's, we get 98% of information being preserved

        # Select the number of components to retain
        # -----------------------------------------
        df_pca_reduced = pd.DataFrame(data = principalComponents,
                                columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5'])

        # ADDING SOME VISUALIZATION
        # Bar graph for explained variance ratio
        plt.bar([1,2,3,4,5], list(ev*100), label='Principal Components', color='b')
        plt.legend()
        plt.xlabel('Principal Components')
        pc=[]
        for i in range(5):
            pc.append('PC'+str(i+1))
        plt.xticks([1,2,3,4,5],pc,fontsize=8,rotation=30)
        plt.ylabel('Variance Ratio')
        plt.title(f'Variance Ratio of S{subject}-E{exercise} Dataset')
        plt.savefig(f'plots/ev/variance-ratio-s{subject}-e{exercise}.png')
        plt.figure()

if __name__ == '__main__':
    main()
