#!/usr/bin/env python3

import constants
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def main():

    exercise = 1
    unit = 2

    # Load dataset for each subject
    for subject in range(1, constants.SUBJECTS + 1):
        df = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test.txt', delimiter=';')
