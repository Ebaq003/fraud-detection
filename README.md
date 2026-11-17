# Credit Card Fraud Detection: Model Comparison

This project compares the performance of two machine learning classification models, **Logistic Regression** and **Decision Tree**, on the **ULB Credit Card Fraud dataset**. Each model is trained with two techniques for handling class imbalance: **SMOTE** and **class weights**.

## Objective

To predict whether a transaction is fraudulent (`0 = Legitimate`, `1 = Fraud`) and to compare the effectiveness of each model based on:

- Precision
- Recall
- PR-AUC (average precision)
- Frauds caught (TP) and false alarms (FP)

Accuracy is not used as a main metric, because only about 0.17% of transactions are fraud, so a model that predicts "legitimate" every time would score about 99.8%.

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- `Time`: seconds since the first transaction
- `V1` to `V28`: anonymized features (PCA components)
- `Amount`: transaction amount
- `Class`: target (`0` or `1`)

The data is not included in this repo because of its size. The download cell in the notebook fetches it automatically, or place `creditcard.csv` in a `data/` folder.

## Models Used

- **Logistic Regression**: Linear classifier with `max_iter=1000`
- **Decision Tree**: Tree-based model with `max_depth=6`

## Imbalance Techniques

- **SMOTE**: Creates synthetic fraud examples in the training data only
- **Class Weights**: Uses `class_weight="balanced"` so fraud mistakes are penalized more

## Preprocessing

- Duplicate rows removed before splitting
- Stratified train/test split
- `Amount` scaled with `RobustScaler` and `Time` with `StandardScaler`, inside the pipeline so the scalers learn from the training data only

## Results

| Model | Precision | Recall | PR-AUC | Frauds caught | False alarms |
|---|---|---|---|---|---|
| Logistic + SMOTE | 0.0544 | 0.8737 | 0.6747 | 83 | 1442 |
| Logistic + Class Weights | 0.0565 | 0.8737 | 0.6721 | 83 | 1386 |
| Tree + SMOTE | 0.0566 | 0.8316 | 0.4597 | 79 | 1316 |
| Tree + Class Weights | 0.0680 | 0.8316 | 0.4691 | 79 | 1082 |


## Conclusion

Among the two models and two imbalance techniques tested:

- **Logistic Regression** performed the best overall:
  - Highest PR-AUC: **0.67**
  - Caught the most frauds: **83 of 95**
- **Decision Tree** ranked fraud less well:
  - PR-AUC of about 0.46-0.47
  - Class weights gave about 230 fewer false alarms than SMOTE for the same frauds caught
- All pipelines have low precision (5-7%) at the default 0.5 threshold.

Based on this analysis, **Logistic Regression** would be the best model for this dataset.

Results come from a single train/test split with 95 fraud cases, so small differences may be due to chance.

## Future Work

- Threshold tuning
- Hyperparameter tuning with cross-validation
- Trying ensemble models like Random Forest or XGBoost

