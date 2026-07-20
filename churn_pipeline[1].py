import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix

RANDOM_STATE = 42
sns.set_style("whitegrid")

# ---------- 1. LOAD ----------
df = pd.read_csv("telco_churn_raw.csv")
print("Raw shape:", df.shape)

# ---------- 2. CLEAN ----------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])  # tenure=0 customers
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("Nulls after clean:", df.isnull().sum().sum())
print("Churn rate: {:.2f}%".format(df["Churn"].mean() * 100))

# ---------- 3. EDA ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Churn by Contract Type")

sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[0, 1])
axes[0, 1].set_title("Tenure vs Churn")

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0])
axes[1, 0].set_title("Monthly Charges vs Churn")

sns.countplot(data=df, x="InternetService", hue="Churn", ax=axes[1, 1])
axes[1, 1].set_title("Churn by Internet Service")

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=130)
plt.close()

# Top churn-driver correlations (numeric + encoded categoricals)
df_enc_for_corr = df.copy()
cat_cols_all = df_enc_for_corr.select_dtypes(include="object").columns
for c in cat_cols_all:
    df_enc_for_corr[c] = LabelEncoder().fit_transform(df_enc_for_corr[c])
corr_with_churn = df_enc_for_corr.corr()["Churn"].drop("Churn").sort_values(key=abs, ascending=False)
top5_drivers = corr_with_churn.head(5)
print("\nTop 5 churn drivers (by correlation):")
print(top5_drivers)

# ---------- 4. FEATURE ENGINEERING ----------
data = df.copy()

# Binary yes/no columns -> 1/0
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for c in binary_cols:
    data[c] = data[c].map({"Yes": 1, "No": 0})

# Multi-category columns needing internet-dependent cleanup
service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
for c in service_cols:
    data[c] = data[c].replace({"No internet service": "No", "No phone service": "No"})

# Engineered features
data["NumServices"] = data[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
data["AvgChargePerMonth"] = data["TotalCharges"] / data["tenure"].replace(0, 1)
data["IsNewCustomer"] = (data["tenure"] <= 6).astype(int)
data["HasMultipleServices"] = (data["NumServices"] >= 3).astype(int)

# One-hot encode remaining categoricals
cat_cols = data.select_dtypes(include="object").columns.tolist()
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

feature_count = data_encoded.shape[1] - 1
print(f"\nTotal engineered features: {feature_count}")

# ---------- 5. TRAIN/TEST SPLIT ----------
X = data_encoded.drop(columns=["Churn"])
y = data_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 6. MODELS ----------
results = {}

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
pred_lr = log_reg.predict(X_test_scaled)
results["Logistic Regression"] = {
    "f1": f1_score(y_test, pred_lr),
    "roc_auc": roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1]),
}

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results["Random Forest"] = {
    "f1": f1_score(y_test, pred_rf),
    "roc_auc": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
}

gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)
results["Gradient Boosting"] = {
    "f1": f1_score(y_test, pred_gb),
    "roc_auc": roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]),
}

print("\n=== MODEL RESULTS ===")
for name, m in results.items():
    print(f"{name}: F1 = {m['f1']:.3f} | ROC-AUC = {m['roc_auc']:.3f}")

best_model_name = max(results, key=lambda k: results[k]["f1"])
best_model = {"Logistic Regression": log_reg, "Random Forest": rf, "Gradient Boosting": gb}[best_model_name]
best_pred = {"Logistic Regression": pred_lr, "Random Forest": pred_rf, "Gradient Boosting": pred_gb}[best_model_name]
print(f"\nBest model: {best_model_name} (F1={results[best_model_name]['f1']:.3f})")
print("\nClassification report (best model):")
print(classification_report(y_test, best_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=130)
plt.close()

# Feature importance (if tree-based best model, else RF for reference)
importance_model = rf
importances = pd.Series(importance_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=130)
plt.close()

# ---------- 7. K-MEANS SEGMENTATION ----------
cluster_features = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
cluster_data = data[cluster_features].copy()
cluster_scaler = StandardScaler()
cluster_scaled = cluster_scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
data["Cluster"] = kmeans.fit_predict(cluster_scaled)

cluster_summary = data.groupby("Cluster").agg(
    customers=("Churn", "count"),
    churn_rate=("Churn", "mean"),
    avg_tenure=("tenure", "mean"),
    avg_monthly_charge=("MonthlyCharges", "mean"),
    avg_services=("NumServices", "mean"),
).round(2)
cluster_summary["churn_rate"] = (cluster_summary["churn_rate"] * 100).round(1)
print("\n=== CUSTOMER SEGMENTS (K-MEANS, k=4) ===")
print(cluster_summary)

highest_risk_cluster = cluster_summary["churn_rate"].idxmax()
highest_risk_pct_of_base = (cluster_summary.loc[highest_risk_cluster, "customers"] / len(data) * 100)
print(f"\nHighest-risk cluster: {highest_risk_cluster} "
      f"({cluster_summary.loc[highest_risk_cluster,'churn_rate']}% churn rate, "
      f"{highest_risk_pct_of_base:.1f}% of customer base)")

plt.figure(figsize=(7, 5))
sns.scatterplot(data=data, x="tenure", y="MonthlyCharges", hue="Cluster", palette="Set2", alpha=0.7)
plt.title("Customer Segments — Tenure vs Monthly Charges")
plt.tight_layout()
plt.savefig("customer_segments.png", dpi=130)
plt.close()

# ---------- 8. SAVE SUMMARY ----------
with open("results_summary.txt", "w") as f:
    f.write("CUSTOMER CHURN PREDICTION - RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} real customer records (IBM Telco Customer Churn dataset)\n")
    f.write(f"Churn rate: {df['Churn'].mean()*100:.2f}%\n")
    f.write(f"Engineered features: {feature_count}\n\n")
    f.write("Top 5 Churn Drivers (correlation with churn):\n")
    for feat, val in top5_drivers.items():
        f.write(f"  - {feat}: {val:.3f}\n")
    f.write("\nModel Performance:\n")
    for name, m in results.items():
        f.write(f"  - {name}: F1 = {m['f1']:.3f}, ROC-AUC = {m['roc_auc']:.3f}\n")
    f.write(f"\nBest Model: {best_model_name}\n\n")
    f.write("Customer Segments (K-Means, k=4):\n")
    f.write(cluster_summary.to_string())
    f.write(f"\n\nHighest-risk segment: Cluster {highest_risk_cluster} "
            f"({cluster_summary.loc[highest_risk_cluster,'churn_rate']}% churn rate)\n")

print("\nSaved: eda_overview.png, confusion_matrix.png, feature_importance.png, customer_segments.png, results_summary.txt")
