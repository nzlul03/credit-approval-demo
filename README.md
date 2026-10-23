# Credit Approval Prediction

An interactive demo page: enter an applicant profile and get a loan approval prediction from a Logistic Regression model.

**Live demo:** https://nzlul03.github.io/credit-approval-demo/

## How it works

- The model is a scikit-learn pipeline: `StandardScaler` on six numeric features, `OneHotEncoder` on employment type, then `LogisticRegression`.
- Its parameters are exported to `model.json` and embedded in the page. Prediction runs entirely in the browser (`p = 1 / (1 + e^-z)`), with no backend and no data sent anywhere.
- The "most influential factors" bars show each feature's contribution to the logit relative to an average training applicant.

## Files

| File | Purpose |
|---|---|
| `index.html` | Built page, served by GitHub Pages |
| `template.html` | Page source, with a `/*MODEL*/` placeholder |
| `model.json` | Exported model parameters |
| `train.py` | Retrains the model and writes `model.json` (needs the original dataset, not included) |
| `build.py` | Injects `model.json` into `template.html` and writes `index.html` |

## Disclaimer

This is a portfolio demo trained on a small sample dataset. It is not a real credit decision.

Author: Nuzulul Khairu Nissa
