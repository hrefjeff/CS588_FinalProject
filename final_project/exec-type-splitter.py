#!/usr/bin/env python3

import pandas as pd

def main():
    subject = 1
    exercise = 1
    unit = 2

    df = pd.read_csv(f's{subject}/e{exercise}/u{unit}/test.txt', delimiter=';')
    extracted_df = df.iloc[100:1900]
    extracted_df.to_csv(f's{subject}/e{exercise}/u{unit}/test-correct.csv', sep=';', index=False)

    extracted_df = df.iloc[4200:5700]
    extracted_df.to_csv(f's{subject}/e{exercise}/u{unit}/test-toolow.csv', sep=';', index=False)

if __name__ == '__main__':
    main()
