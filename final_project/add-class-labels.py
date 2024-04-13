#!/usr/bin/env python3

import pandas as pd

# Function to read the dataset
def load_data(filename):
    data = pd.read_csv(filename, delimiter=';')
    return data

# Function to add labels based on specified time intervals
def label_data(data, intervals, labels):
    """
    Args:
    data: DataFrame containing the sensor data.
    intervals: List of tuples, where each tuple contains the start and end index of the interval.
    labels: List of labels corresponding to each interval.

    Returns:
    Modified DataFrame with a new 'label' column.
    """
    # Create a new column for labels and initially set to None or a default label
    data['label'] = None
    for interval, label in zip(intervals, labels):
        # Apply the label within the specified interval
        data.loc[interval[0]:interval[1], 'label'] = label
    return data

# Main script to load data, define intervals, and apply labels
if __name__ == "__main__":

    subject = 1
    exercise = 1
    unit = 2

    infile = f's{subject}/e{exercise}/u{unit}/test.txt'
    outfile = f's{subject}/e{exercise}/u{unit}/test-labeled.csv'
    data = load_data(infile)

    # Define your intervals and labels here
    # For example, intervals as [(start_index1, end_index1), (start_index2, end_index2), ...]
    # Labels corresponding to these intervals
    intervals = [(0, 2000), (2001, 2700), (2401, 3300), (3301, 4100), (4101, 5949)]
    labels = [1, 0, 2, 0, 3]

    # Label the data
    labeled_data = label_data(data, intervals, labels)

    # Optionally, save the labeled data to a new CSV file
    labeled_data.to_csv(outfile, sep=';', index=False)

    # Show the labeled part of the data
    print(labeled_data.head(10))
