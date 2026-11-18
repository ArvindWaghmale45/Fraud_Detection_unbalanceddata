# EDA Summary

The ULB credit card fraud dataset contains 284,807 transactions and 31 columns:
Time, V1-V28, Amount and Class. It has no missing values.

## Class imbalance

- Genuine: 284,315 transactions (99.8273%)
- Fraud: 492 transactions (0.1727%)
- Fraud ratio: approximately 1 fraud per 579 transactions

## Split and model result

The notebook used an 80/20 stratified split with random state 42. The test set
contains 56,962 transactions and 98 fraud cases. Class weights were used for
imbalance handling. The chosen Random Forest produced 81 true positives, 17
false negatives and 17 false positives on the test set, with 0.8265 precision,
0.8265 recall and 0.8265 F1.

The tuned threshold was 0.4954. V14, V10, V12, V17 and V4 were the top global
features. Amount had small global importance.
