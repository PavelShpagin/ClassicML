"""
Ensemble of multiple submissions to achieve better score.
Based on the example in README.
"""

import pandas as pd
import numpy as np

# Read submissions
sub1 = pd.read_csv("submission_simple.csv")  # Our 0.216 score submission
sub2 = pd.read_csv("submission_geometric.csv")  # Geometric mean baseline
sub3 = pd.read_csv("submission_seasonality.csv")  # With December boost

# Check if IDs match
assert all(sub1["id"] == sub2["id"]), "IDs do not match!"
assert all(sub1["id"] == sub3["id"]), "IDs do not match!"

# Weighted ensemble
# Give more weight to the simple one that scored 0.216
# and combine with geometric mean approaches
sub_ens = sub1.copy()
sub_ens["new_house_transaction_amount"] = (
    0.4 * sub1["new_house_transaction_amount"] +  # Conservative approach that scored
    0.35 * sub2["new_house_transaction_amount"] +  # Geometric mean
    0.25 * sub3["new_house_transaction_amount"]    # Seasonality
)

# Save ensemble
sub_ens.to_csv("submission_ensemble.csv", index=False)

print("✅ Ensemble submission created: submission_ensemble.csv")
print(f"  Mean: {sub_ens['new_house_transaction_amount'].mean():.2f}")
print(f"  Std: {sub_ens['new_house_transaction_amount'].std():.2f}")
print(f"  Non-zero: {(sub_ens['new_house_transaction_amount'] > 0).sum()} / {len(sub_ens)}")
print(f"  Max: {sub_ens['new_house_transaction_amount'].max():.2f}")

# Also create a more aggressive ensemble focusing on geometric mean
sub_ens2 = sub1.copy()
sub_ens2["new_house_transaction_amount"] = (
    0.2 * sub1["new_house_transaction_amount"] +  # Conservative 
    0.5 * sub2["new_house_transaction_amount"] +  # Geometric mean (higher weight)
    0.3 * sub3["new_house_transaction_amount"]    # Seasonality
)

sub_ens2.to_csv("submission_ensemble_v2.csv", index=False)
print("\n✅ Alternative ensemble created: submission_ensemble_v2.csv")
print(f"  Mean: {sub_ens2['new_house_transaction_amount'].mean():.2f}")
print(f"  Non-zero: {(sub_ens2['new_house_transaction_amount'] > 0).sum()} / {len(sub_ens2)}")

