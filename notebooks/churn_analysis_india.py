"""
Customer Churn Prediction & Retention Analytics - Indian Telecom Market
========================================================================
A comprehensive machine learning project for predicting customer churn
in the Indian telecommunications industry.

Author: Data Analytics Team
Date: January 2026
Context: Indian Telecom Market with INR currency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("CUSTOMER CHURN PREDICTION - INDIAN TELECOM MARKET")
print("="*70)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

print("\n" + "="*70)
print("SECTION 1: DATA LOADING AND EXPLORATION")
print("="*70)

# Load data
df = pd.read_csv('./data/customer_churn_data.csv')

print(f"\n✓ Indian Telecom Dataset loaded successfully")
print(f"  - Total customers: {len(df):,}")
print(f"  - Total features: {len(df.columns)}")

# Basic info
print("\n📊 Dataset Overview:")
print(f"  - Shape: {df.shape}")
print(f"  - Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\n⚠️  Missing values found:")
    print(missing[missing > 0])
else:
    print(f"\n✓ No missing values detected")

# Churn rate
churn_rate = df['Churned'].mean() * 100
print(f"\n📈 Key Metrics:")
print(f"  - Overall Churn Rate: {churn_rate:.1f}%")
print(f"  - Churned Customers: {df['Churned'].sum():,}")
print(f"  - Active Customers: {(df['Churned'] == 0).sum():,}")
print(f"  - Prepaid Customers: {(df['Plan_Type'] == 'Prepaid').sum():,} ({(df['Plan_Type'] == 'Prepaid').mean()*100:.1f}%)")
print(f"  - Postpaid Customers: {(df['Plan_Type'] == 'Postpaid').sum():,} ({(df['Plan_Type'] == 'Postpaid').mean()*100:.1f}%)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*70)
print("SECTION 2: EXPLORATORY DATA ANALYSIS - INDIAN MARKET")
print("="*70)

# Churn by Plan Type
print("\n📊 Churn Rate by Plan Type:")
churn_by_plan = df.groupby('Plan_Type')['Churned'].agg(['sum', 'count', 'mean'])
churn_by_plan.columns = ['Churned', 'Total', 'Churn_Rate']
churn_by_plan['Churn_Rate'] = churn_by_plan['Churn_Rate'] * 100
churn_by_plan = churn_by_plan.sort_values('Churn_Rate', ascending=False)
print(churn_by_plan.to_string())

# Churn by Contract Type
print("\n📊 Churn Rate by Contract Type:")
churn_by_contract = df.groupby('Contract_Type')['Churned'].agg(['sum', 'count', 'mean'])
churn_by_contract.columns = ['Churned', 'Total', 'Churn_Rate']
churn_by_contract['Churn_Rate'] = churn_by_contract['Churn_Rate'] * 100
churn_by_contract = churn_by_contract.sort_values('Churn_Rate', ascending=False)
print(churn_by_contract.to_string())

# Churn by City Tier
print("\n📊 Churn Rate by City Tier:")
churn_by_tier = df.groupby('City_Tier')['Churned'].agg(['sum', 'count', 'mean'])
churn_by_tier.columns = ['Churned', 'Total', 'Churn_Rate']
churn_by_tier['Churn_Rate'] = churn_by_tier['Churn_Rate'] * 100
print(churn_by_tier.to_string())

# Churn by Tenure
print("\n📊 Churn Rate by Tenure Groups:")
df['Tenure_Group'] = pd.cut(df['Tenure_Months'], 
                             bins=[0, 6, 12, 24, 48, 100],
                             labels=['0-6 months', '6-12 months', '1-2 years', '2-4 years', '4+ years'])
churn_by_tenure = df.groupby('Tenure_Group')['Churned'].agg(['sum', 'count', 'mean'])
churn_by_tenure.columns = ['Churned', 'Total', 'Churn_Rate']
churn_by_tenure['Churn_Rate'] = churn_by_tenure['Churn_Rate'] * 100
print(churn_by_tenure.to_string())

# Churn by Network Quality
print("\n📊 Churn Rate by Network Quality:")
churn_by_network = df.groupby('Network_Quality')['Churned'].agg(['sum', 'count', 'mean'])
churn_by_network.columns = ['Churned', 'Total', 'Churn_Rate']
churn_by_network['Churn_Rate'] = churn_by_network['Churn_Rate'] * 100
print(churn_by_network.to_string())

# Monthly charges analysis
print("\n📊 Monthly Charges Analysis (INR):")
print(f"  Churned customers avg: ₹{df[df['Churned']==1]['Monthly_Charges_INR'].mean():,.2f}")
print(f"  Active customers avg: ₹{df[df['Churned']==0]['Monthly_Charges_INR'].mean():,.2f}")

# Support calls analysis
print("\n📊 Support Calls Analysis:")
print(f"  Churned customers avg: {df[df['Churned']==1]['Num_Support_Calls'].mean():.1f} calls")
print(f"  Active customers avg: {df[df['Churned']==0]['Num_Support_Calls'].mean():.1f} calls")

# Top 5 states by churn
print("\n📊 Top 5 States by Total Customers:")
print(df['State'].value_counts().head().to_string())

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("SECTION 3: DATA PREPROCESSING")
print("="*70)

# Create a copy for modeling
df_model = df.copy()

# Drop CustomerID and temporary columns
df_model = df_model.drop(['CustomerID', 'Tenure_Group'], axis=1)

# Encode categorical variables
print("\n🔄 Encoding categorical variables...")
label_encoders = {}
categorical_cols = df_model.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

print(f"  ✓ Encoded {len(categorical_cols)} categorical columns")

# Feature Engineering
print("\n🔧 Engineering new features...")

# Interaction features
df_model['Charges_Per_Tenure'] = df_model['Monthly_Charges_INR'] / (df_model['Tenure_Months'] + 1)
df_model['Total_Services'] = (df_model[['Voice_Service', 'Internet_Service', 
                                         'SMS_Pack', 'Data_Rollover']].sum(axis=1))
df_model['Support_Per_Tenure'] = df_model['Num_Support_Calls'] / (df_model['Tenure_Months'] + 1)
df_model['Is_New_Customer'] = (df_model['Tenure_Months'] <= 3).astype(int)
df_model['High_Value_Customer'] = (df_model['Monthly_Charges_INR'] > df_model['Monthly_Charges_INR'].median()).astype(int)
df_model['Has_Premium_OTT'] = ((df_model['OTT_Subscriptions'] != 4) & (df_model['OTT_Subscriptions'] != 0)).astype(int)
df_model['Network_Quality_Low'] = (df_model['Network_Quality'] <= 2).astype(int)

print(f"  ✓ Created 7 new engineered features")

# Separate features and target
X = df_model.drop('Churned', axis=1)
y = df_model['Churned']

print(f"\n📋 Final feature set:")
print(f"  - Total features: {X.shape[1]}")
print(f"  - Target variable: Churned (0=Active, 1=Churned)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\n✂️  Data split:")
print(f"  - Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  - Training churn rate: {y_train.mean()*100:.1f}%")
print(f"  - Test churn rate: {y_test.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n  ✓ Features scaled using StandardScaler")

# ============================================================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n" + "="*70)
print("SECTION 4: MODEL TRAINING AND EVALUATION")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n🤖 Training models...\n")

for name, model in models.items():
    print(f"{'─'*70}")
    print(f"Training: {name}")
    print(f"{'─'*70}")
    
    # Train
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  - Accuracy: {accuracy*100:.2f}%")
    print(f"  - ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 Confusion Matrix:")
    print(f"  - True Negatives: {tn:,}")
    print(f"  - False Positives: {fp:,}")
    print(f"  - False Negatives: {fn:,}")
    print(f"  - True Positives: {tp:,}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 Additional Metrics:")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1:.4f}")
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Select best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']

print(f"\n{'='*70}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")
print(f"{'='*70}")

# ============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SECTION 5: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

if best_model_name != 'Logistic Regression':
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🎯 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
else:
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': np.abs(best_model.coef_[0])
    }).sort_values('Coefficient', ascending=False)
    
    print(f"\n🎯 Top 10 Most Important Features:")
    print(coefficients.head(10).to_string(index=False))

# ============================================================================
# 6. CUSTOMER RISK SEGMENTATION
# ============================================================================

print("\n" + "="*70)
print("SECTION 6: CUSTOMER RISK SEGMENTATION")
print("="*70)

# Get predictions for all customers
if best_model_name == 'Logistic Regression':
    df['Churn_Probability'] = best_model.predict_proba(scaler.transform(X))[:, 1]
else:
    df['Churn_Probability'] = best_model.predict_proba(X)[:, 1]

# Create risk segments
df['Risk_Segment'] = pd.cut(df['Churn_Probability'], 
                             bins=[0, 0.3, 0.6, 1.0],
                             labels=['Low Risk', 'Medium Risk', 'High Risk'])

print(f"\n🎯 Customer Risk Distribution:")
risk_distribution = df['Risk_Segment'].value_counts().sort_index()
for segment, count in risk_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"  - {segment}: {count:,} customers ({percentage:.1f}%)")

# High-risk customer profile
high_risk = df[df['Risk_Segment'] == 'High Risk']

print(f"\n⚠️  High-Risk Customer Profile:")
print(f"  - Count: {len(high_risk):,}")
print(f"  - Avg Monthly Charges: ₹{high_risk['Monthly_Charges_INR'].mean():,.2f}")
print(f"  - Avg Tenure: {high_risk['Tenure_Months'].mean():.1f} months")
print(f"  - Avg Support Calls: {high_risk['Num_Support_Calls'].mean():.1f}")
print(f"  - Avg Network Quality: {high_risk['Network_Quality'].mean():.2f}/5")
print(f"  - Avg Satisfaction: {high_risk['Satisfaction_Score'].mean():.2f}/5")
print(f"  - Prepaid %: {(high_risk['Plan_Type']=='Prepaid').mean()*100:.1f}%")

# ============================================================================
# 7. BUSINESS IMPACT ANALYSIS - INDIAN MARKET
# ============================================================================

print("\n" + "="*70)
print("SECTION 7: BUSINESS IMPACT ANALYSIS (INR)")
print("="*70)

# Calculate potential revenue at risk
avg_customer_lifetime_value = df['Total_Charges_INR'].mean()
high_risk_revenue = len(high_risk) * avg_customer_lifetime_value

print(f"\n💰 Revenue Impact (Indian Rupees):")
print(f"  - Average Customer Lifetime Value: ₹{avg_customer_lifetime_value:,.2f}")
print(f"  - Revenue at Risk (High Risk customers): ₹{high_risk_revenue:,.2f}")
print(f"  - Revenue at Risk (Lakhs): ₹{high_risk_revenue/100000:.2f} L")
print(f"  - Revenue at Risk (Crores): ₹{high_risk_revenue/10000000:.2f} Cr")
print(f"  - Monthly Revenue at Risk: ₹{high_risk['Monthly_Charges_INR'].sum():,.2f}")

# If we can reduce churn by 20% in high-risk segment
saved_customers = len(high_risk) * 0.20 * results[best_model_name]['precision']
saved_revenue = saved_customers * avg_customer_lifetime_value

print(f"\n📈 Potential Impact of Retention Campaign:")
print(f"  - Assumption: 20% reduction in high-risk churn")
print(f"  - Estimated customers saved: {saved_customers:.0f}")
print(f"  - Estimated revenue saved: ₹{saved_revenue:,.2f}")
print(f"  - Estimated revenue saved (Lakhs): ₹{saved_revenue/100000:.2f} L")
print(f"  - Estimated revenue saved (Crores): ₹{saved_revenue/10000000:.2f} Cr")

# Campaign budget recommendation
campaign_cost_per_customer = 1500  # INR
total_campaign_budget = len(high_risk) * campaign_cost_per_customer
roi_percentage = (saved_revenue / total_campaign_budget - 1) * 100

print(f"\n💡 Campaign Budget Analysis:")
print(f"  - Recommended spend per customer: ₹{campaign_cost_per_customer:,}")
print(f"  - Total campaign budget: ₹{total_campaign_budget:,.2f}")
print(f"  - Expected ROI: {roi_percentage:.1f}%")

# ============================================================================
# 8. ACTIONABLE RECOMMENDATIONS - INDIAN TELECOM
# ============================================================================

print("\n" + "="*70)
print("SECTION 8: ACTIONABLE RECOMMENDATIONS - INDIAN MARKET")
print("="*70)

print("\n🎯 RETENTION STRATEGY RECOMMENDATIONS:\n")

print("1. IMMEDIATE ACTION - High-Risk Customers:")
print("   •", f"Target the {len(high_risk):,} high-risk customers immediately")
print("   • Offer plan upgrades with cashback (₹500-1000)")
print("   • Provide complimentary OTT subscriptions (Disney+ Hotstar/Prime)")
print("   • Free data rollover for 3 months")
print("   • Priority customer support with dedicated helpline")

print("\n2. PREPAID TO POSTPAID CONVERSION:")
print("   • Focus on high-value prepaid customers")
print("   • Offer postpaid migration benefits (extra data, OTT free)")
print("   • No security deposit for good credit history")
print("   • Bill payment reminders via WhatsApp/SMS")

print("\n3. NETWORK QUALITY IMPROVEMENT:")
print("   • Address network quality complaints in Tier 2/3 cities")
print("   • Proactive communication during network maintenance")
print("   • Compensation for poor network experience")
print("   • Speed test guarantees with recharge credits")

print("\n4. VALUE-ADDED SERVICES:")
print("   • Bundle OTT platforms (Netflix, Prime, Hotstar)")
print("   • Partner with regional OTT platforms")
print("   • Music streaming integration (Gaana, JioSaavn)")
print("   • Gaming partnerships for youth segment")

print("\n5. PAYMENT METHOD OPTIMIZATION:")
print("   • Promote UPI auto-pay with cashback")
print("   • Integrate with popular UPI apps (PhonePe, Paytm, GPay)")
print("   • Instant recharge through WhatsApp")
print("   • Reward points for timely payments")

print("\n6. REGIONAL STRATEGY:")
print("   • State-specific offers and local language support")
print("   • Regional festival offers (Diwali, Pongal, Durga Puja)")
print("   • Partnership with local brands")
print("   • City-tier specific pricing")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SECTION 9: SAVING RESULTS")
print("="*70)

# Save predictions
output_df = df[['CustomerID', 'Churned', 'Churn_Probability', 'Risk_Segment', 
                'State', 'City_Tier', 'Plan_Type']].copy()
output_df.to_csv('./data/customer_churn_predictions.csv', index=False)
print(f"\n✓ Predictions saved to: customer_churn_predictions.csv")

# Save high-risk customers for action
high_risk_action = df[df['Risk_Segment'] == 'High Risk'][
    ['CustomerID', 'Churn_Probability', 'Monthly_Charges_INR', 'State', 'City_Tier',
     'Tenure_Months', 'Plan_Type', 'Contract_Type', 'Num_Support_Calls', 
     'Network_Quality', 'Satisfaction_Score']
].sort_values('Churn_Probability', ascending=False)

high_risk_action.to_csv('./data/high_risk_customers_action_list.csv', index=False)
print(f"✓ High-risk action list saved to: high_risk_customers_action_list.csv")

# Save model performance
performance_summary = pd.DataFrame(results).T
performance_summary = performance_summary[['accuracy', 'roc_auc', 'precision', 'recall', 'f1']]
performance_summary.to_csv('./data/model_performance.csv')
print(f"✓ Model performance saved to: model_performance.csv")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"\nKey Deliverables (Indian Telecom Market):")
print(f"  1. Churn prediction model (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
print(f"  2. {len(high_risk):,} high-risk customers identified")
print(f"  3. ₹{saved_revenue/10000000:.2f} Cr potential revenue saved")
print(f"  4. Actionable retention strategy for Indian market")
print("\n")
