import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=563) # oh so this guy does all the behind the scenes ha🤣🤣

# print head of data, targets and features
print(f'features are : \n{dataset.data.features}\n targets are : \n{dataset.data.targets}')
X = dataset.data.features
y = dataset.data.targets.iloc[:, 0] # convert the df to series. nice try...

print("Shape:", X.shape)
print("Target distribution:")
print(y.value_counts(normalize=True))
print(dataset.variables)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y, # split with same class proportions.
)

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
probabilities = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

fpr, tpr, thresholds = roc_curve(y_test, probabilities)
auc = roc_auc_score(y_test, probabilities)
artifact_dir = Path("ml_platform/training/churn/models")
artifact_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
plt.savefig(artifact_dir)


joblib.dump(
    pipeline,
    artifact_dir / "churn_pipeline.joblib",
)

(artifact_dir / "feature_names.json").write_text(
    json.dumps(X.columns.tolist(), indent=2)
)
