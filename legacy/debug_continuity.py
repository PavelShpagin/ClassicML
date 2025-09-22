import pandas as pd
import numpy as np

# Load training data
train = pd.read_csv('data/raw/train/new_house_transactions.csv')

# Parse month to numeric
def parse_month(m):
    parts = m.replace('-', ' ').split()
    year = int(parts[0])
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    month = month_map[parts[1]]
    return year * 12 + month

train['time'] = train['month'].apply(parse_month)
train['sector_id'] = train['sector'].str.extract('(\d+)').astype(int)

# Check continuity
unique_times = sorted(train['time'].unique())
print("Unique time points in training data:")
print(f"Total: {len(unique_times)}")
print(f"Min: {min(unique_times)} Max: {max(unique_times)}")
print(f"Expected range: {max(unique_times) - min(unique_times) + 1}")

# Convert back to readable
def time_to_month(t):
    year = (t - 1) // 12
    month = (t - 1) % 12 + 1
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return f"{year}-{month_names[month-1]}"

print("\nActual months in training data (in order):")
for t in unique_times:
    print(f"  {t}: {time_to_month(t)}")

# Check test months
test = pd.read_csv('data/raw/test.csv')
test_months = sorted(test['id'].str.split('_').str[0].unique())

print("\n\nTest months to predict:")
for tm in test_months:
    t = parse_month(tm.replace(' ', '-'))
    print(f"  {t}: {tm}")

print("\n\n=== KEY FINDING ===")
print(f"Last available training time: {max(unique_times)} = {time_to_month(max(unique_times))}")
print(f"First test time: {parse_month(test_months[0].replace(' ', '-'))} = {test_months[0]}")
print(f"Gap: {parse_month(test_months[0].replace(' ', '-')) - max(unique_times)} months")

# Check how our code maps these
print("\n\n=== OUR CODE'S TIME MAPPING ===")
# Reproduce what our code does
train['time_code'] = train['time'] - min(unique_times)
print(f"Our code maps time {min(unique_times)} -> 0")
print(f"Our code maps time {max(unique_times)} -> {max(unique_times) - min(unique_times)}")

# What time index would test months get?
print("\nTest months in our code's indexing:")
for tm in test_months:
    actual_time = parse_month(tm.replace(' ', '-'))
    code_time = actual_time - min(unique_times)
    print(f"  {tm}: actual_time={actual_time}, code_time={code_time}")
