import pandas as pd
import numpy as np
from pathlib import Path

# Load training data
train = pd.read_csv('data/raw/train/new_house_transactions.csv')
print("Training data shape:", train.shape)
print("\nFirst 5 rows of train:")
print(train.head())
print("\nLast 5 rows of train:")
print(train.tail())

# Check unique months
months = sorted(train['month'].unique())
print(f"\nTotal unique months in train: {len(months)}")
print("First 3 months:", months[:3])
print("Last 3 months:", months[-3:])

# Load test data
test = pd.read_csv('data/raw/test.csv')
print("\n\nTest data shape:", test.shape)
print("\nFirst 5 rows of test:")
print(test.head())

# Extract test months
test_months = sorted(test['id'].str.split('_').str[0].unique())
print(f"\nTest months to predict: {test_months}")

# Check the gap
print("\n\n=== TIME GAP ANALYSIS ===")
print(f"Last training month: {months[-1]}")
print(f"Test months: {test_months}")

# Parse to understand the gap
def parse_month(m):
    parts = m.replace('-', ' ').split()
    year = int(parts[0])
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    month = month_map[parts[1]]
    return year * 12 + month

last_train_time = parse_month(months[-1])
first_test_time = parse_month(test_months[0])
print(f"\nGap between last train and first test: {first_test_time - last_train_time} months")
