"""
train.py
--------
Full ML pipeline for Salary Prediction System
  1. Load & explore dataset
  2. Preprocessing (encoding, scaling, feature engineering)
  3. Train 3 models: Linear Regression, Random Forest, Gradient Boosting
  4. Compare & evaluate all models (R², MAE, RMSE, CV scores)
  5. Save best model + preprocessor + metrics to /models/
  6. Save evaluation plots

Run:  python3 train.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing   import StandardScaler, OrdinalEncoder
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import (mean_absolute_error, mean_squared_error,
                                     r2_score, mean_absolute_percentage_error)

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "salary_dataset.csv")
MODELS = os.path.join(BASE, "models")
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(PLOTS,  exist_ok=True)

print("=" * 65)
print("  SALARY PREDICTION — ML TRAINING PIPELINE")
print("=" * 65)

print("\n[1/6] Loading dataset…")
df = pd.read_csv(DATA)
print(f"  Shape : {df.shape}")
print(f"  Target: salary_lpa  |  min={df.salary_lpa.min():.1f}  max={df.salary_lpa.max():.1f}  mean={df.salary_lpa.mean():.1f}")
print(f"  Nulls : {df.isnull().sum().sum()}")

TARGET = "salary_lpa"
DROP   = ["increment_pct", "promotion_prob"]

X = df.drop(columns=[TARGET] + DROP)
y = df[TARGET]

feature_names = X.columns.tolist()

CAT_COLS = ["job_role", "industry", "education_level", "location", "company_type", "company_size"]
NUM_COLS = [c for c in X.columns if c not in CAT_COLS]

print(f"  Numeric features  ({len(NUM_COLS)}): {NUM_COLS}")
print(f"  Categorical features ({len(CAT_COLS)}): {CAT_COLS}")

print("\n[2/6] Splitting data (80/20)…")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

print("\n[3/6] Building preprocessing pipeline…")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     NUM_COLS),
    ("cat", categorical_transformer, CAT_COLS),
])

preprocessor.fit(X_train)
X_train_prep = preprocessor.transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

print(f"  Transformed shape: {X_train_prep.shape[1]} features after encoding")
joblib.dump(preprocessor, os.path.join(MODELS, "preprocessor.pkl"))
print("  Preprocessor saved ✓")

print("\n[4/6] Training models…")

models = {
    "Linear Regression": Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, max_depth=5, subsample=0.8, min_samples_leaf=5, random_state=42),
}

results  = {}
trained  = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  ── {name} ──")
    model.fit(X_train_prep, y_train)
    trained[name] = model

    y_pred = model.predict(X_test_prep)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    cv_r2 = cross_val_score(model, X_train_prep, y_train, cv=kf, scoring="r2", n_jobs=-1)

    results[name] = {
        "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 2), "CV_R2_mean": round(cv_r2.mean(), 4),
        "CV_R2_std": round(cv_r2.std(), 4), "y_pred": y_pred,
    }
    print(f"    R²    : {r2:.4f}")
    print(f"    MAE   : {mae:.2f} LPA")
    print(f"    RMSE  : {rmse:.2f} LPA")
    print(f"    MAPE  : {mape:.1f}%")
    print(f"    CV R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

print("\n[5/6] Selecting & saving best model…")
best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = trained[best_name]
joblib.dump(best_model, os.path.join(MODELS, "best_model.pkl"))
print(f"  Best model: {best_name}  (R² = {results[best_name]['R2']})")

for name, model in trained.items():
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODELS, fname))

metrics_export = {
    k: {mk: mv for mk, mv in v.items() if mk != "y_pred"}
    for k, v in results.items()
}
metrics_export["best_model"] = best_name
metrics_export["feature_names"] = feature_names
metrics_export["num_cols"] = NUM_COLS
metrics_export["cat_cols"] = CAT_COLS

rf_model = trained["Random Forest"]
ohe_cats = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(CAT_COLS).tolist()
all_feature_names = NUM_COLS + ohe_cats
importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
top15 = importances.nlargest(15).to_dict()
metrics_export["top15_features"] = {k: round(v, 5) for k, v in top15.items()}

with open(os.path.join(MODELS, "metrics.json"), "w") as f:
    json.dump(metrics_export, f, indent=2)
print("  Metrics saved ✓")

print("\n[6/6] Generating evaluation plots…")

DARK = "#0f1520"; ACCENT = "#00d4ff"; PURPLE = "#7c3aed"; GREEN = "#10b981"
TEXT = "#e8edf5"; MUTED  = "#6b7a8d"; COLORS = [ACCENT, PURPLE, GREEN]

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK, "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#1e2d3d", "font.family": "monospace",
})

model_names = list(results.keys())

# Plot 1: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", color=TEXT)
for ax, mkey, mlabel, color in zip(axes, ["R2","MAE","RMSE"], ["R²","MAE ₹LPA","RMSE ₹LPA"], COLORS):
    vals = [results[m][mkey] for m in model_names]
    bars = ax.barh(model_names, vals, color=color, alpha=0.85, height=0.5)
    ax.set_xlabel(mlabel, fontsize=9); ax.set_title(mkey, fontsize=11, fontweight="bold"); ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8.5, color=TEXT)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "01_model_comparison.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Plot 1 ✓")

# Plot 2: Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Actual vs Predicted — All Models", fontsize=14, fontweight="bold", color=TEXT)
for ax, (name, color) in zip(axes, zip(model_names, COLORS)):
    y_pred = results[name]["y_pred"]
    ax.scatter(y_test, y_pred, alpha=0.25, s=8, color=color)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "--", color="white", alpha=0.5, lw=1.5)
    ax.set_xlabel("Actual (₹ LPA)"); ax.set_ylabel("Predicted (₹ LPA)")
    ax.set_title(f"{name}\nR²={results[name]['R2']:.3f}", fontsize=10, fontweight="bold"); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "02_actual_vs_predicted.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Plot 2 ✓")

# Plot 3: Residuals
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Residual Distributions", fontsize=14, fontweight="bold", color=TEXT)
for ax, (name, color) in zip(axes, zip(model_names, COLORS)):
    residuals = y_test.values - results[name]["y_pred"]
    ax.hist(residuals, bins=60, color=color, alpha=0.8, edgecolor="none")
    ax.axvline(0, color="white", linestyle="--", alpha=0.7, lw=1.5)
    ax.set_xlabel("Residual (₹ LPA)"); ax.set_ylabel("Count"); ax.set_title(name, fontsize=10, fontweight="bold"); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "03_residuals.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Plot 3 ✓")

# Plot 4: Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))
top = importances.nlargest(15).sort_values()
colors_imp = [PURPLE if i < 5 else ACCENT if i < 10 else GREEN for i in range(len(top))]
bars = ax.barh(top.index, top.values, color=colors_imp, alpha=0.85, height=0.6)
ax.set_xlabel("Feature Importance (Random Forest)"); ax.set_title("Top 15 Most Influential Features", fontsize=13, fontweight="bold"); ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, top.values):
    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=7.5, color=TEXT)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "04_feature_importance.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Plot 4 ✓")

# Plot 5: Salary Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Salary Distribution in Dataset", fontsize=13, fontweight="bold", color=TEXT)
axes[0].hist(y, bins=80, color=ACCENT, alpha=0.8, edgecolor="none")
axes[0].set_xlabel("Salary (₹ LPA)"); axes[0].set_ylabel("Count"); axes[0].set_title("Overall Distribution", fontsize=10, fontweight="bold"); axes[0].grid(alpha=0.2)
top_roles = df.groupby("job_role")["salary_lpa"].median().nlargest(10)
axes[1].barh(top_roles.index, top_roles.values, color=PURPLE, alpha=0.85, height=0.6)
axes[1].set_xlabel("Median Salary ₹ LPA"); axes[1].set_title("Median Salary by Top 10 Roles", fontsize=10, fontweight="bold"); axes[1].grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "05_salary_distribution.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Plot 5 ✓")

# Plot 6: CV Scores
fig, ax = plt.subplots(figsize=(9, 5))
means = [results[m]["CV_R2_mean"] for m in model_names]
stds  = [results[m]["CV_R2_std"]  for m in model_names]
x = np.arange(len(model_names))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=COLORS, alpha=0.85, error_kw={"ecolor": TEXT, "elinewidth": 1.5})
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("5-Fold CV R² Score"); ax.set_title("Cross-Validation Scores (Mean ± Std)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center", fontsize=10, color=TEXT, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "06_cv_scores.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  Plot 6 ✓")

print("\n" + "="*65)
print(f"  DONE — Best Model: {best_name}  R²={results[best_name]['R2']}")
print(f"  Run:  streamlit run app.py")
print("="*65)
