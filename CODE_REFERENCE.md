#  Complete Code Reference Guide

##  Quick Navigation

This document explains what code does what in the Customer Churn Prediction project.

---

##       **Dataset Generation Code**

**File:** `data/generate_indian_data.py`

### What it does:
Creates a realistic 5,000 customer dataset for Indian telecom market

### Key Code Sections:

**Lines 1-20: Import libraries and setup**
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
np.random.seed(42)  # For reproducibility
```

**Lines 21-50: Customer Demographics**
```python
# Creates Indian customer profiles
states = ['Maharashtra', 'Karnataka', 'Delhi', 'Tamil Nadu'...]
city_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
ages = np.random.normal(35, 13, n_customers)
```

**Lines 51-90: Telecom Services**
```python
# Indian telecom specific
plan_types = ['Prepaid', 'Postpaid']  # 70% prepaid
internet_service = ['4G', '5G', 'No Internet']
ott_subscriptions = ['Netflix', 'Prime Video', 'Disney+ Hotstar']
payment_method = ['UPI', 'Net Banking', 'Debit Card', 'Credit Card']
```

**Lines 91-120: Monthly Charges (INR)**
```python
base_charge = 150  # Base in rupees
monthly_charges += np.where(plan_types == 'Postpaid', 200, 0)
monthly_charges += np.where(internet_service == '5G', 500, 0)
# Final range: ₹150 - ₹2500
```

**Lines 121-145: Churn Probability Calculation       CRITICAL**
```python
# Factors that INCREASE churn
churn_prob += np.where(plan_types == 'Prepaid', 0.20, 0)
churn_prob += np.where(tenure_months < 6, 0.28, 0)
churn_prob += np.where(network_quality <= 2, 0.25, 0)

# Factors that DECREASE churn
churn_prob -= np.where(tenure_months > 24, 0.22, 0)
churn_prob -= np.where(ott_subscriptions == 'Multiple', 0.15, 0)

# Generate actual churn
churned = np.random.binomial(1, churn_prob)
```

**Lines 146-180: Save Dataset**
```python
df = pd.DataFrame({...})
df.to_csv('customer_churn_data.csv', index=False)
```

### How to Run:
```bash
cd data/
python generate_indian_data.py
```

**Output:** `customer_churn_data.csv` (5,000 rows × 26 columns)

---

##   **Machine Learning Pipeline Code**                

**File:** `notebooks/churn_analysis_india.py`

### What it does:
Complete ML pipeline - trains models, makes predictions, creates risk segments

### Key Code Sections:

#### **SECTION 1: Data Loading (Lines 1-50)**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('customer_churn_data.csv')
print(f"Total customers: {len(df):,}")
print(f"Churn rate: {df['Churned'].mean()*100:.1f}%")
```

#### **SECTION 2: EDA (Lines 51-120)**
```python
# Analyzes patterns
churn_by_plan = df.groupby('Plan_Type')['Churned'].mean()
# Prepaid: 32.9%, Postpaid: 21.9%

churn_by_tier = df.groupby('City_Tier')['Churned'].mean()
# Tier 3: 36.2% (highest)

churn_by_tenure = df.groupby('Tenure_Group')['Churned'].mean()
# 0-6 months: 53.8% (new customers at risk!)
```

#### **SECTION 3: Preprocessing (Lines 121-180)**
```python
# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# Feature Engineering      
df_model['Charges_Per_Tenure'] = Monthly_Charges / (Tenure + 1)
df_model['Is_New_Customer'] = (Tenure <= 3).astype(int)
df_model['High_Value_Customer'] = (Monthly_Charges > median).astype(int)
df_model['Network_Quality_Low'] = (Network_Quality <= 2).astype(int)
df_model['Has_Premium_OTT'] = (has OTT).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 4,000 training, 1,000 testing

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### **SECTION 4: Model Training (Lines 181-280)**       PREDICTIONS START HERE
```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    # Train model
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results[name] = {
        'model': model,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }

# Select best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
# Result: Gradient Boosting with 81.9% ROC-AUC
```

#### **SECTION 5: Feature Importance (Lines 281-310)**
```python
# Shows what matters most
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Top features:
# 1. Total_Charges_INR
# 2. Tenure_Months
# 3. Satisfaction_Score
# 4. Contract_Type
# 5. Network_Quality
```

#### **SECTION 6: Risk Segmentation (Lines 311-380)**       CREATES PREDICTIONS
```python
# Apply model to ALL 5,000 customers
if best_model_name == 'Logistic Regression':
    df['Churn_Probability'] = best_model.predict_proba(
        scaler.transform(X)
    )[:, 1]
else:
    df['Churn_Probability'] = best_model.predict_proba(X)[:, 1]

# Each customer now has probability (0.0 to 1.0)
# Example: Customer A = 0.78 (78% likely to churn)

# Create risk segments
df['Risk_Segment'] = pd.cut(
    df['Churn_Probability'], 
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Results:
# Low Risk: 3,145 customers (62.9%)
# Medium Risk: 1,131 customers (22.6%)
# High Risk: 724 customers (14.5%) ← TARGET THESE!

# High-risk profile
high_risk = df[df['Risk_Segment'] == 'High Risk']
print(f"Avg monthly charges: ₹{high_risk['Monthly_Charges_INR'].mean():.2f}")
print(f"Avg tenure: {high_risk['Tenure_Months'].mean():.1f} months")
print(f"Prepaid %: {(high_risk['Plan_Type']=='Prepaid').mean()*100:.1f}%")
```

#### **SECTION 7: Business Impact (Lines 381-430)**
```python
# Calculate revenue at risk
avg_customer_lifetime_value = df['Total_Charges_INR'].mean()
# ₹19,979 per customer

high_risk_revenue = len(high_risk) * avg_customer_lifetime_value
# ₹14,464,915 = ₹1.45 Crores at risk

# Campaign impact
saved_customers = len(high_risk) * 0.20 * precision
# If we reduce churn by 20%: ~101 customers saved

saved_revenue = saved_customers * avg_customer_lifetime_value
# ₹2,025,088 = ₹20.25 Lakhs potential savings

# ROI calculation
campaign_budget = len(high_risk) * 1500  # ₹1,500 per customer
roi = (saved_revenue / campaign_budget - 1) * 100
# 86.5% ROI
```

#### **SECTION 8: Recommendations (Lines 431-480)**
```python
print("RETENTION STRATEGY:")
print("1. Target 724 high-risk customers")
print("2. Offer ₹500-1000 cashback on annual plans")
print("3. Free Disney+ Hotstar or Prime Video")
print("4. UPI auto-pay with ₹200 cashback")
print("5. Priority customer support")
```

#### **SECTION 9: Save Results (Lines 481-520)**       CREATES OUTPUT FILES
```python
# Save all predictions
output_df = df[[
    'CustomerID', 'Churned', 'Churn_Probability', 'Risk_Segment',
    'State', 'City_Tier', 'Plan_Type'
]]
output_df.to_csv('customer_churn_predictions.csv', index=False)

# Save high-risk customers for immediate action
high_risk_action = df[df['Risk_Segment'] == 'High Risk'][[
    'CustomerID', 'Churn_Probability', 'Monthly_Charges_INR',
    'State', 'City_Tier', 'Tenure_Months', 'Plan_Type',
    'Num_Support_Calls', 'Network_Quality', 'Satisfaction_Score'
]].sort_values('Churn_Probability', ascending=False)

high_risk_action.to_csv('high_risk_customers_action_list.csv', index=False)

# Save model performance
performance_summary = pd.DataFrame(results).T
performance_summary.to_csv('model_performance.csv')
```

### How to Run:
```bash
cd notebooks/
python churn_analysis_india.py
```

**Outputs Created:**
1. `customer_churn_predictions.csv` - All 5,000 with risk scores
2. `high_risk_customers_action_list.csv` - 724 priority customers
3. `model_performance.csv` - Model comparison metrics

---

##   **Dashboard Code**

**File:** `dashboards/churn_dashboard.html`

### What it does:
Interactive HTML dashboard with charts (no server needed!)

### Key Code Sections:

#### **Lines 1-140: HTML Structure & CSS**
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { 
            background: linear-gradient(135deg, #065A82, #1C7293);
            font-family: 'Segoe UI';
        }
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 3em;
            font-weight: bold;
            color: #F96167;  /* Red for danger metrics */
        }
    </style>
</head>
```

#### **Lines 141-200: Metric Cards**
```html
<div class="metrics-grid">
    <div class="metric-card danger">
        <div class="metric-label">Overall Churn Rate</div>
        <div class="metric-value">29.7%</div>
        <div class="metric-description">1,484 of 5,000 customers</div>
    </div>
    
    <div class="metric-card warning">
        <div class="metric-label">High-Risk Customers</div>
        <div class="metric-value">724</div>
    </div>
    
    <div class="metric-card info">
        <div class="metric-label">Revenue at Risk</div>
        <div class="metric-value">₹1.45 Cr</div>
    </div>
    
    <div class="metric-card success">
        <div class="metric-label">Potential Savings</div>
        <div class="metric-value">₹20.25 L</div>
    </div>
</div>
```

#### **Lines 201-249: Chart Containers**
```html
<div class="chart-card">
    <div class="chart-title">Churn Rate by Plan Type</div>
    <div id="contractChart"></div>  <!-- Chart renders here -->
</div>

<div class="chart-card">
    <div class="chart-title">Churn Rate by City Tier</div>
    <div id="tenureChart"></div>
</div>

<div id="riskChart"></div>          <!-- Pie chart -->
<div id="satisfactionChart"></div>  <!-- Line chart -->
<div id="modelChart"></div>         <!-- Bar chart -->
```

#### **Lines 250-350: Insights Section**
```html
<div class="insights-section">
    <h2>Key Insights & Recommendations - Indian Telecom</h2>
    
    <div class="insight-item">
        <div class="insight-heading">1. Prepaid Churn Risk</div>
        <div class="insight-text">
            Prepaid customers churn at 32.9%...
            <strong>Action:</strong> Incentivize prepaid-to-postpaid
        </div>
    </div>
    
    <!-- 5 more insights... -->
</div>
```

#### **Lines 351-450: JavaScript Charts**       CREATES VISUALIZATIONS
```javascript
<script>
// Chart 1: Plan Type Churn
var contractData = [{
    x: ['Prepaid', 'Postpaid'],
    y: [32.86, 21.88],  // From analysis results
    type: 'bar',
    marker: { color: ['#F96167', '#2C5F2D'] },
    text: ['32.9%', '21.9%'],
    textposition: 'outside'
}];

var contractLayout = {
    yaxis: { title: 'Churn Rate (%)' },
    xaxis: { title: 'Plan Type' },
    height: 350
};

Plotly.newPlot('contractChart', contractData, contractLayout, {responsive: true});

// Chart 2: City Tier
var tenureData = [{
    x: ['Tier 1', 'Tier 2', 'Tier 3'],
    y: [26.55, 28.20, 36.20],
    type: 'bar',
    marker: { color: '#1C7293' }
}];
Plotly.newPlot('tenureChart', tenureData, tenureLayout);

// Chart 3: Risk Segmentation Pie Chart
var riskData = [{
    values: [724, 1131, 3145],
    labels: ['High Risk', 'Medium Risk', 'Low Risk'],
    type: 'pie',
    marker: { colors: ['#F96167', '#F9E795', '#2C5F2D'] },
    textinfo: 'label+percent'
}];
Plotly.newPlot('riskChart', riskData, riskLayout);

// Chart 4: Network Quality Line Chart
var satisfactionData = [{
    x: ['Quality 1', 'Quality 2', 'Quality 3', 'Quality 4', 'Quality 5'],
    y: [50.00, 51.06, 29.29, 24.75, 24.24],
    type: 'scatter',
    mode: 'lines+markers',
    marker: { color: '#065A82', size: 12 },
    line: { width: 3 }
}];
Plotly.newPlot('satisfactionChart', satisfactionData, satisfactionLayout);

// Chart 5: Model Performance Comparison
var modelData = [
    {
        x: ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        y: [0.7905, 0.7977, 0.8186],
        type: 'bar',
        name: 'ROC-AUC Score',
        marker: { color: '#1C7293' }
    },
    {
        x: ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        y: [0.7770, 0.7700, 0.7790],
        type: 'bar',
        name: 'Accuracy',
        marker: { color: '#065A82' }
    }
];

var modelLayout = {
    yaxis: { title: 'Score', range: [0, 1] },
    barmode: 'group',
    showlegend: true
};
Plotly.newPlot('modelChart', modelData, modelLayout);
</script>
```

### How to Use:
```bash
# Just open in browser - no installation needed!
open churn_dashboard.html
```

**Features:**
-        Fully interactive charts (zoom, pan, hover)
-        Responsive design (works on mobile)
-        No server required
-        All data embedded in HTML

---

##   **Presentation Code**

**File:** `reports/create_presentation.py`

### What it does:
Generates PowerPoint presentation using python-pptx library

### Key Code Sections:

#### **Lines 1-50: Setup & Configuration**
```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# Load predictions
df = pd.read_csv('customer_churn_predictions.csv')

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(5.625)

# Color palette - Ocean theme
NAVY = RGBColor(6, 90, 130)
TEAL = RGBColor(28, 114, 147)
WHITE = RGBColor(255, 255, 255)
```

#### **Lines 51-150: Slide Creation Functions**
```python
def add_title_slide(prs, title, subtitle):
    """Creates title slide with dark background"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Add navy background
    background = slide.shapes.add_shape(1, 0, 0, width, height)
    background.fill.solid()
    background.fill.fore_color.rgb = NAVY
    
    # Add title text
    title_box = slide.shapes.add_textbox(...)
    title_frame.text = title
    title_para.font.size = Pt(44)
    title_para.font.color.rgb = WHITE

def add_content_slide(prs, title, content_func):
    """Creates content slide with white background"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Add title bar
    title_bar = slide.shapes.add_shape(...)
    title_bar.fill.fore_color.rgb = NAVY
    
    # Call content function
    content_func(slide)
```

#### **Lines 151-250: Slide 1 - Title**
```python
add_title_slide(prs, 
    "Customer Churn Prediction & Retention Analytics",
    "Data-Driven Strategy for Indian Telecom Market"
)
```

#### **Lines 251-350: Slide 2 - Executive Summary**
```python
def executive_summary(slide):
    # Create stat boxes
    stats = [
        ("29.7%", "Overall Churn Rate", x=1.5),
        ("724", "High-Risk Customers", x=4.5),
        ("₹20.25 L", "Potential Savings", x=7.5)
    ]
    
    for value, label, x_pos in stats:
        # Create colored box
        stat_box = slide.shapes.add_shape(...)
        stat_box.fill.fore_color.rgb = TEAL
        
        # Add large value
        value_box.text = value
        value_para.font.size = Pt(42)
        value_para.font.color.rgb = WHITE
        
        # Add label below
        label_box.text = label
        
    # Add key insight
    insight_box.text = "By targeting 724 high-risk customers..."

add_content_slide(prs, "Executive Summary", executive_summary)
```

#### **Lines 351-450: Slides 3-6 - Analysis, Model, Recommendations, ROI**
```python
def churn_analysis(slide):
    # Left column: Churn drivers
    # Right column: Risk segments
    
def model_performance(slide):
    # Model metrics and top features
    
def recommendations(slide):
    # India-specific retention strategies
    
def roi_slide(slide):
    # Financial impact and timeline
```

#### **Lines 451-500: Save Presentation**
```python
# Save final presentation
output_path = 'Customer_Churn_Analytics_Presentation.pptx'
prs.save(output_path)
print(f"       Presentation created: {output_path}")
```

### How to Run:
```bash
cd reports/
python create_presentation.py
```

**Output:** `Customer_Churn_Analytics_Presentation.pptx` (7 slides)

---

##         **Summary Table**

| File | Lines | Purpose | Output |
|------|-------|---------|--------|
| `generate_indian_data.py` | 180 | Creates dataset | `customer_churn_data.csv` |
| `churn_analysis_india.py` | 520 | ML pipeline | 3 CSV files with predictions |
| `churn_dashboard.html` | 450 | Interactive dashboard | Opens in browser |
| `create_presentation.py` | 500 | PowerPoint | `.pptx` file |

---

##          **Running the Complete Pipeline**

```bash
# Step 1: Generate data
cd data/
python generate_indian_data.py

# Step 2: Run analysis and create predictions
cd ../notebooks/
python churn_analysis_india.py

# Step 3: Create presentation
cd ../reports/
python create_presentation.py

# Step 4: View dashboard
cd ../dashboards/
open churn_dashboard.html  # or double-click file
```

---

##           **Key Code Locations**

### Where is the ML model trained?
→ `notebooks/churn_analysis_india.py` lines 181-280

### Where are predictions created?
→ `notebooks/churn_analysis_india.py` lines 311-380

### Where are charts created?
→ `dashboards/churn_dashboard.html` lines 351-450

### Where is churn probability calculated?
→ `data/generate_indian_data.py` lines 121-145

### Where is business impact calculated?
→ `notebooks/churn_analysis_india.py` lines 381-430

---

**Now you have the complete code reference!**  
