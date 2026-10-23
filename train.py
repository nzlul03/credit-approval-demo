"""Retrain the notebook's Logistic Regression and export model.json for the landing page.
Run from anywhere:  python3 landing-page/train.py
Needs: pandas, scikit-learn. Reads ../credit_approval.csv, writes model.json next to this file."""
import json, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

here = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(here.parent / "credit_approval.csv")
num = ["age", "monthly_income", "credit_score", "existing_loans", "loan_amount", "loan_term_months"]
cat = ["employment_type"]
X, y = df[num + cat], df["approved"]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([("prep", pre), ("model", LogisticRegression())]).fit(Xtr, ytr)
pred = pipe.predict(Xte)
m = pipe.named_steps["model"]
sc = pre.named_transformers_["num"]
oh = pre.named_transformers_["cat"]
freq = df.employment_type.value_counts(normalize=True)
out = dict(
    num=num, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
    cats=list(oh.categories_[0]), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]),
    acc=accuracy_score(yte, pred), cm=confusion_matrix(yte, pred, labels=["No", "Yes"]).tolist(),
    n=len(df), rng={c: [float(df[c].min()), float(df[c].max())] for c in num},
    freq=[float(freq[c]) for c in oh.categories_[0]], approval_rate=float((y == "Yes").mean()),
)
(here / "model.json").write_text(json.dumps(out))
print("test accuracy:", out["acc"], "-> wrote model.json")
for a in [dict(age=28, employment_type="Permanent", monthly_income=9.5, credit_score=720, existing_loans=0, loan_amount=60, loan_term_months=36),
          dict(age=18, employment_type="Self-employed", monthly_income=2.5, credit_score=540, existing_loans=3, loan_amount=20, loan_term_months=12)]:
    print("sanity P(approved) =", round(pipe.predict_proba(pd.DataFrame([a]))[0][1], 3))
