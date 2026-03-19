
# TELECOM CUSTOMER CHURN PREDICTION
# Complete Analysis Pipeline
# Author: Prabin Pokhrel


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score, roc_curve)

sns.set_theme(style="whitegrid")


# CONFIGURATION


DATA_PATH   = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\Churn-prediction\data\Telco_customer_churn.xlsx'
OUTPUT_PATH = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\Churn-prediction\output'
os.makedirs(OUTPUT_PATH, exist_ok=True)


# STEP 1: LOAD DATA


print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")



# STEP 2: CLEAN DATA


print("\n" + "=" * 60)
print("STEP 2: CLEANING DATA")
print("=" * 60)

# Fix Total Charges
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)

# Create numeric churn column
df['Churn_Num'] = df['Churn Value'].astype(int)

# Standardise Senior Citizen
df['Senior Citizen'] = df['Senior Citizen'].map({1:'Yes', 0:'No'})

print(f"✅ Cleaned: {df.shape[0]:,} rows remaining")
print(f"📊 Overall Churn Rate: {df['Churn_Num'].mean()*100:.1f}%")
print(f"📊 Churned customers: {df['Churn_Num'].sum():,}")
print(f"📊 Retained customers: {(df['Churn_Num']==0).sum():,}")

# Save cleaned data for Power BI
df.to_csv(os.path.join(OUTPUT_PATH, 'churn_cleaned.csv'),
          index=False, encoding='utf-8-sig')
print("✅ churn_cleaned.csv saved!")


# STEP 3: EXPLORATORY ANALYSIS & CHARTS


print("\n" + "=" * 60)
print("STEP 3: ANALYSIS & VISUALISATIONS")
print("=" * 60)

# Chart 1: Churn Distribution 
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['Churn Label'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=['#2ECC71', '#E74C3C'], edgecolor='white', width=0.5)
ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Churn Status', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 50,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_churn_distribution.png'), dpi=150)
plt.show()
print("✅ Chart 1 saved!")


# Chart 2: Churn Rate by Contract Type 
contract = df.groupby('Contract')['Churn_Num'].mean() * 100
contract = contract.sort_values(ascending=False)
print(f"\n📊 Churn by Contract:\n{contract.round(1)}")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(contract.index, contract.values,
              color=['#E74C3C', '#E67E22', '#2ECC71'],
              edgecolor='white', width=0.5)
ax.set_title('Churn Rate by Contract Type', fontsize=14,
             fontweight='bold', pad=15)
ax.set_xlabel('Contract Type', fontsize=12)
ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_ylim(0, 55)
for bar, val in zip(bars, contract.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center',
            fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_contract_churn.png'), dpi=150)
plt.show()
print("✅ Chart 2 saved!")


# Chart 3: Monthly Charges vs Churn 
print(f"\n📊 Avg Monthly Charges:")
print(df.groupby('Churn Label')['Monthly Charges'].mean().round(2))

fig, ax = plt.subplots(figsize=(7, 5))
retained = df[df['Churn Label']=='No']['Monthly Charges']
churned  = df[df['Churn Label']=='Yes']['Monthly Charges']
bp = ax.boxplot([retained, churned],
                labels=['Retained', 'Churned'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#3498DB')
bp['boxes'][1].set_facecolor('#E74C3C')
ax.set_title('Monthly Charges: Retained vs Churned',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Churn Status', fontsize=12)
ax.set_ylabel('Monthly Charges ($)', fontsize=12)
ax.text(1, retained.median(), f'  Median: ${retained.median():.0f}',
        va='center', fontsize=10, color='#3498DB')
ax.text(2, churned.median(), f'  Median: ${churned.median():.0f}',
        va='center', fontsize=10, color='#E74C3C')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_charges_churn.png'), dpi=150)
plt.show()
print("✅ Chart 3 saved!")


# Chart 4: Tenure vs Churn (side by side) 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Customer Tenure vs Churn', fontsize=14,
             fontweight='bold', y=1.01)

axes[0].hist(df[df['Churn Label']=='Yes']['Tenure Months'],
             bins=24, color='#E74C3C', edgecolor='white', alpha=0.9)
axes[0].set_title('Churned Customers', fontsize=12)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Number of Customers')
axes[0].set_xlim(0, 75)
axes[0].annotate('Most churn\nhappens here',
                  xy=(3, 550), fontsize=9,
                  color='darkred',
                  arrowprops=dict(arrowstyle='->', color='darkred'),
                  xytext=(15, 500))

axes[1].hist(df[df['Churn Label']=='No']['Tenure Months'],
             bins=24, color='#2ECC71', edgecolor='white', alpha=0.9)
axes[1].set_title('Retained Customers', fontsize=12)
axes[1].set_xlabel('Tenure (months)')
axes[1].set_ylabel('Number of Customers')
axes[1].set_xlim(0, 75)
axes[1].annotate('Most loyal\ncustomers here',
                  xy=(70, 580), fontsize=9,
                  color='darkgreen',
                  arrowprops=dict(arrowstyle='->', color='darkgreen'),
                  xytext=(45, 520))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '04_tenure_churn.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 4 saved!")


# Chart 5: Churn by Internet Service
internet = df.groupby('Internet Service')['Churn_Num'].mean() * 100
internet = internet.sort_values(ascending=False)
print(f"\n📊 Churn by Internet Service:\n{internet.round(1)}")

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(internet.index, internet.values,
              color=['#E74C3C', '#3498DB', '#2ECC71'],
              edgecolor='white', width=0.5)
ax.set_title('Churn Rate by Internet Service', fontsize=14,
             fontweight='bold', pad=15)
ax.set_xlabel('Internet Service Type', fontsize=12)
ax.set_ylabel('Churn Rate (%)', fontsize=12)
ax.set_ylim(0, 55)
for bar, val in zip(bars, internet.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center',
            fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '05_internet_churn.png'), dpi=150)
plt.show()
print("✅ Chart 5 saved!")



# STEP 4: MACHINE LEARNING MODELS


print("\n" + "=" * 60)
print("STEP 4: MACHINE LEARNING MODELS")
print("=" * 60)

#  4.1 Prepare features 
features = [
    'Tenure Months', 'Monthly Charges', 'Total Charges',
    'Contract', 'Internet Service', 'Payment Method',
    'Tech Support', 'Online Security', 'Online Backup',
    'Device Protection', 'Streaming TV', 'Streaming Movies',
    'Paperless Billing', 'Senior Citizen', 'Partner', 'Dependents'
]

# Keep original text for Power BI export
df_powerbi = df[features + ['Churn_Num', 'Churn Label']].copy()

# Encode for ML
df_model = df_powerbi.copy()
le = LabelEncoder()
text_cols = df_model.select_dtypes(include='object').columns.tolist()
text_cols = [c for c in text_cols if c != 'Churn Label']

for col in text_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

#  4.2 Split
X = df_model[features]
y = df_model['Churn_Num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"✅ Training set : {X_train.shape[0]:,} rows")
print(f"✅ Testing set  : {X_test.shape[0]:,} rows")

# 4.3 Logistic Regression 
print("\n⏳ Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_acc  = accuracy_score(y_test, lr_pred) * 100
lr_auc  = roc_auc_score(y_test, lr_prob)
print(f"✅ Accuracy: {lr_acc:.1f}% | AUC: {lr_auc:.3f}")
print(classification_report(y_test, lr_pred,
      target_names=['Retained', 'Churned']))

# 4.4 Random Forest
print("\n⏳ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_acc  = accuracy_score(y_test, rf_pred) * 100
rf_auc  = roc_auc_score(y_test, rf_prob)
print(f"✅ Accuracy: {rf_acc:.1f}% | AUC: {rf_auc:.3f}")
print(classification_report(y_test, rf_pred,
      target_names=['Retained', 'Churned']))


# Chart 6: Model Accuracy Comparison 
fig, ax = plt.subplots(figsize=(7, 5))
models     = ['Logistic Regression', 'Random Forest']
accuracies = [lr_acc, rf_acc]
colors_m   = ['#3498DB', '#2ECC71']
bars = ax.bar(models, accuracies, color=colors_m,
              edgecolor='white', width=0.5)
ax.set_title('Model Accuracy Comparison', fontsize=14,
             fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(70, 90)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center',
            fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '06_model_comparison.png'), dpi=150)
plt.show()
print("✅ Chart 6 saved!")


# Chart 7: ROC Curve 
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lr_fpr, lr_tpr, color='#3498DB', lw=2,
        label=f'Logistic Regression (AUC={lr_auc:.3f})')
ax.plot(rf_fpr, rf_tpr, color='#2ECC71', lw=2,
        label=f'Random Forest (AUC={rf_auc:.3f})')
ax.plot([0,1],[0,1], color='gray', lw=1,
        linestyle='--', label='Random Guess')
ax.fill_between(lr_fpr, lr_tpr, alpha=0.05, color='#3498DB')
ax.fill_between(rf_fpr, rf_tpr, alpha=0.05, color='#2ECC71')
ax.set_title('ROC Curve — Model Comparison', fontsize=14,
             fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '07_roc_curve.png'), dpi=150)
plt.show()
print("✅ Chart 7 saved!")


# Chart 8: Confusion Matrix 
cm = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Retained', 'Churned'])
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Random Forest — Confusion Matrix',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_confusion_matrix.png'), dpi=150)
plt.show()
print("✅ Chart 8 saved!")


# Chart 9: Feature Importance 
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top10    = feat_imp.sort_values(ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top10.index, top10.values,
               color='#3498DB', edgecolor='white')
ax.set_title('Top 10 Churn Predictors (Random Forest)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score', fontsize=12)
for bar, val in zip(bars, top10.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '09_feature_importance.png'), dpi=150)
plt.show()
print("✅ Chart 9 saved!")



# STEP 5: OUTPUTS FOR POWER BI


print("\n" + "=" * 60)
print("STEP 5: SAVING POWER BI FILES")
print("=" * 60)

# Get test set with original text columns
X_test_original = df_powerbi.loc[X_test.index, features].copy()
X_test_original['Churn_Label']        = df.loc[X_test.index, 'Churn Label'].values
X_test_original['Actual_Churn']       = y_test.values
X_test_original['Predicted_Churn']    = rf_pred
X_test_original['Churn_Probability']  = (rf_prob * 100).round(1)
X_test_original['Risk_Level']         = pd.cut(
    X_test_original['Churn_Probability'],
    bins=[0, 30, 60, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

X_test_original.to_csv(
    os.path.join(OUTPUT_PATH, 'churn_predictions.csv'),
    index=False, encoding='utf-8-sig')

print("✅ churn_predictions.csv saved!")
print(f"\nRisk Level distribution:")
print(X_test_original['Risk_Level'].value_counts())
print(f"\nSample predictions:")
print(X_test_original[['Contract', 'Monthly Charges',
                         'Tenure Months', 'Churn_Probability',
                         'Risk_Level']].head(10))

X_test_original['Risk_Level'] = X_test_original['Risk_Level'].cat.add_categories('Unknown').fillna('Unknown')



# FINAL SUMMARY


print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Total Customers Analysed : {len(df):,}")
print(f"Overall Churn Rate       : {df['Churn_Num'].mean()*100:.1f}%")
print(f"Logistic Regression      : Accuracy {lr_acc:.1f}% | AUC {lr_auc:.3f}")
print(f"Random Forest            : Accuracy {rf_acc:.1f}% | AUC {rf_auc:.3f}")
print(f"Top Churn Predictor      : {feat_imp.idxmax()}")
print(f"\nOutput files saved to:")
print(f"  {OUTPUT_PATH}")
print(f"\n🎉 Analysis Complete! Load churn_predictions.csv into Power BI.")

OUTPUT_PATH = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\Churn-prediction\output'
DATA_PATH   = r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\Churn-prediction\data\Telco_customer_churn.xlsx'

# Load & Clean 
df = pd.read_excel(DATA_PATH)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)
df['Churn_Num'] = df['Churn Value'].astype(int)
df['Senior Citizen'] = df['Senior Citizen'].map({1:'Yes', 0:'No'})

# Print ALL column names to check 
print("All columns in dataset:")
print(df.columns.tolist())

# Features
features = [
    'Tenure Months', 'Monthly Charges', 'Total Charges',
    'Contract', 'Internet Service', 'Payment Method',
    'Tech Support', 'Online Security', 'Online Backup',
    'Device Protection', 'Streaming TV', 'Streaming Movies',
    'Paperless Billing', 'Senior Citizen', 'Partner', 'Dependents'
]

# Encode 
df_model = df[features + ['Churn_Num']].copy()
le = LabelEncoder()
text_cols = df_model.select_dtypes(include='object').columns.tolist()
for col in text_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Split
X = df_model[features]
y = df_model['Churn_Num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

# Build result with Customer Details 
result = df.loc[X_test.index].copy()
result['Predicted_Churn']   = rf_pred
result['Churn_Probability'] = (rf_prob * 100).round(1)
result['Risk_Level']        = pd.cut(
    result['Churn_Probability'],
    bins=[0, 30, 60, 100],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Remove null rows 
result = result[result['Churn_Probability'] > 0]
result = result.dropna(subset=['Risk_Level'])

# Keep only useful columns
result = result[[
    'CustomerID',
    'Gender',
    'Senior Citizen',
    'Partner',
    'Dependents',
    'Tenure Months',
    'Contract',
    'Internet Service',
    'Tech Support',
    'Online Security',
    'Monthly Charges',
    'Total Charges',
    'Payment Method',
    'Churn Label',
    'Predicted_Churn',
    'Churn_Probability',
    'Risk_Level'
]]

# Save 
save_path = os.path.join(OUTPUT_PATH, 'churn_predictions.csv')
result.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n✅ Saved successfully!")
print(f"Total rows     : {len(result)}")
print(f"High Risk      : {len(result[result['Risk_Level']=='High Risk'])}")
print(f"Medium Risk    : {len(result[result['Risk_Level']=='Medium Risk'])}")
print(f"Low Risk       : {len(result[result['Risk_Level']=='Low Risk'])}")
print(f"\nSample High Risk customers:")
print(result[result['Risk_Level']=='High Risk'][[
    'CustomerID','Gender','Contract',
    'Tenure Months','Monthly Charges',
    'Churn_Probability','Risk_Level'
]].head(5).to_string())
import pandas as pd

# Check what columns are in your saved predictions file
df = pd.read_csv(r'C:\Users\WELCOME\Desktop\DataAnalysis_Projects\Churn-prediction\output\churn_predictions.csv')
print("Columns in churn_predictions.csv:")
print(df.columns.tolist())
print(f"\nFirst 3 rows:")
print(df.head(3))