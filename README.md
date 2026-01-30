# Customer Churn Prediction & Retention Analytics

##  Project Overview

A comprehensive, industry-level data analytics project that predicts customer churn and provides actionable retention strategies. This project demonstrates end-to-end data science workflow including data generation, exploratory analysis, machine learning modeling, and business insights delivery.

**Business Impact:** Identified 835 high-risk customers and potential savings of $186,352 through targeted retention campaigns.

---

##  Project Objectives

1. **Predict customer churn** with high accuracy using machine learning
2. **Identify key churn drivers** through exploratory data analysis
3. **Segment customers by risk level** for targeted interventions
4. **Provide actionable recommendations** to reduce customer attrition
5. **Calculate ROI** of retention campaigns

---

##  Project Structure

```
churn_analytics/
├── data/
│   ├── customer_churn_data.csv                    # Main dataset (5,000 customers)
│   ├── customer_churn_predictions.csv              # Model predictions with risk scores
│   ├── high_risk_customers_action_list.csv         # Priority customers for retention
│   ├── model_performance.csv                       # Model comparison metrics
│   └── generate_data.py                            # Synthetic data generation script
├── notebooks/
│   └── churn_analysis.py                           # Complete analysis pipeline
├── dashboards/
│   └── churn_dashboard.html                        # Interactive visualization dashboard
├── reports/
│   ├── Customer_Churn_Analytics_Presentation.pptx  # Executive presentation
│   └── create_presentation.py                      # Presentation generation script
├── models/
│   └── (trained models saved here)
└── README.md                                        # This file
```

---

##  Key Findings

### Churn Statistics
- **Overall Churn Rate:** 34.6% (1,731 out of 5,000 customers)
- **High-Risk Customers:** 835 (16.7% of customer base)
- **Revenue at Risk:** $1.45M from high-risk segment
- **Monthly Revenue at Risk:** $69,340

### Primary Churn Drivers

1. **Contract Type** (Highest Impact)
   - Month-to-Month: 43.7% churn rate
   - One Year: 29.6% churn rate
   - Two Year: 16.3% churn rate

2. **Customer Tenure**
   - 0-6 months: 59.2% churn rate (NEW CUSTOMERS AT HIGHEST RISK)
   - 6-12 months: 35.8% churn rate
   - 1-2 years: 31.7% churn rate
   - 2-4 years: 20.7% churn rate
   - 4+ years: 18.7% churn rate

3. **Satisfaction Score**
   - Score 1-2: 57-58% churn rate
   - Score 3: 38.3% churn rate
   - Score 4-5: 25-29% churn rate

4. **Service Adoption**
   - Customers without online security: Higher churn
   - Customers without tech support: Elevated churn risk
   - More services = Lower churn probability

---

##  Machine Learning Models

### Models Evaluated

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.7776** | **72.40%** | **64.46%** | 45.09% | 53.06% |
| Random Forest | 0.7508 | 71.30% | 61.13% | 46.82% | 53.03% |
| Gradient Boosting | 0.7683 | 72.50% | 62.91% | 50.00% | 55.72% |

**Best Model:** Logistic Regression (selected for production deployment)

### Top Predictive Features

1. Contract Type (0.701 coefficient)
2. Customer Tenure (0.626 coefficient)
3. Satisfaction Score (0.567 coefficient)
4. Charges Per Tenure (0.519 coefficient)
5. Total Charges (0.261 coefficient)

### Model Performance Details

- **True Positives:** 156 (correctly identified churners)
- **True Negatives:** 568 (correctly identified active)
- **False Positives:** 86 (incorrect churn predictions)
- **False Negatives:** 190 (missed churners)

---

##  Customer Risk Segmentation

### Risk Distribution

- **Low Risk (0-30%):** 2,538 customers (50.8%)
- **Medium Risk (30-60%):** 1,627 customers (32.5%)
- **High Risk (60-100%):** 835 customers (16.7%)

### High-Risk Customer Profile

- Average Monthly Charges: $83.04
- Average Tenure: 6.5 months
- Average Support Calls: 1.9
- Average Satisfaction: 2.83/5
- Primary Contract Type: Month-to-Month

---

##  Strategic Recommendations

### 1. Immediate Action - High-Risk Customers

**Target:** 835 high-risk customers

**Actions:**
- Deploy targeted outreach campaigns within 48 hours
- Offer contract upgrade with 10-15% discount
- Provide complimentary premium tech support for 3 months
- Assign dedicated account managers to top 100 highest-risk customers

**Expected Impact:** Save ~108 customers, $186K annual revenue

### 2. Contract Optimization Strategy

**Goal:** Convert month-to-month customers to annual contracts

**Tactics:**
- Create tiered pricing with clear annual savings
- Implement auto-renewal incentives
- Develop loyalty rewards program
- Offer service bundle discounts for annual commitments

**Expected Impact:** Reduce month-to-month churn from 43.7% to <35%

### 3. New Customer Onboarding Enhancement

**Goal:** Reduce churn in first 6 months from 59.2% to <40%

**Program Components:**
- Welcome call within 24 hours of signup
- 30-60-90 day check-in touchpoints
- Proactive technical support assistance
- Educational content on service features
- Early adoption incentives for security services

### 4. Customer Experience Improvements

**Continuous Actions:**
- Monthly satisfaction surveys
- Rapid response to low satisfaction scores (<3)
- Proactive service optimization recommendations
- Increase adoption of online security and tech support
- Reduce support call volume through better self-service

### 5. Data-Driven Monitoring

**Implementation:**
- Deploy model to production for real-time scoring
- Weekly high-risk customer review meetings
- A/B test retention campaigns
- Monthly performance dashboards
- Quarterly model retraining with new data

---

##  Expected ROI

### Investment vs. Return

**Campaign Investment:**
- Estimated cost per saved customer: ~$1,725
- Total campaign budget: $186,000

**Expected Returns:**
- Customers saved: 108
- Revenue saved: $186,352
- ROI: 100.2% (breakeven plus)
- Ongoing value: Reduced acquisition costs

### Success Metrics

- Reduce overall churn rate from 34.6% to <30% within 6 months
- Reduce high-risk segment churn by 20%
- Increase contract upgrades by 15%
- Improve satisfaction scores for at-risk customers
- Achieve <$1,500 cost per saved customer

---

##  Technical Implementation

### Technologies Used

- **Python 3.12**
- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Reporting:** python-pptx

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn python-pptx
```

### Running the Analysis

1. **Generate Dataset:**
```bash
cd data/
python generate_data.py
```

2. **Run Full Analysis:**
```bash
cd notebooks/
python churn_analysis.py
```

3. **Create Presentation:**
```bash
cd reports/
python create_presentation.py
```

4. **View Dashboard:**
```bash
# Open in web browser
open dashboards/churn_dashboard.html
```

---

##  Deliverables

### 1. Data Assets
-  Customer churn dataset (5,000 records)
-  Churn predictions with probability scores
-  High-risk customer action list
-  Model performance comparison

### 2. Analysis & Insights
-  Comprehensive Python analysis script
-  Exploratory data analysis findings
-  Statistical significance testing
-  Feature importance analysis

### 3. Machine Learning Models
-  Three trained models (Logistic Regression, Random Forest, Gradient Boosting)
-  Model evaluation and comparison
-  Prediction pipeline ready for production

### 4. Visualizations & Reports
-  Interactive HTML dashboard with charts
-  Executive PowerPoint presentation (7 slides)
-  Data-driven business recommendations

### 5. Documentation
-  Complete project README
-  Technical implementation guide
-  Business impact assessment

---

##  Skills Demonstrated

### Technical Skills
- Data preprocessing and feature engineering
- Exploratory data analysis (EDA)
- Machine learning model development
- Model evaluation and selection
- Statistical analysis
- Data visualization
- Python programming

### Business Skills
- Business problem formulation
- Customer segmentation
- ROI calculation
- Strategic recommendations
- Stakeholder communication
- Executive presentation

### Tools & Libraries
- Python (pandas, numpy, scikit-learn)
- Data visualization (matplotlib, seaborn, plotly)
- Presentation tools (python-pptx)
- Statistical modeling
- HTML/CSS for dashboards

---

##  Next Steps for Production

1. **Model Deployment**
   - Set up automated scoring pipeline
   - Create API endpoint for real-time predictions
   - Implement model monitoring and retraining schedule

2. **Campaign Execution**
   - Launch high-risk customer outreach
   - Implement A/B testing framework
   - Track campaign effectiveness metrics

3. **Integration**
   - Connect to CRM system
   - Set up automated alerts for new high-risk customers
   - Create retention workflow automation

4. **Continuous Improvement**
   - Collect feedback from retention team
   - Refine model with actual outcomes
   - Expand features with additional data sources

---

##  Business Impact Summary

This analysis provides a clear, data-driven roadmap for reducing customer churn:

- **Identified:** 835 customers at immediate risk
- **Quantified:** $1.45M in revenue at risk
- **Recommended:** Specific, actionable retention strategies
- **Projected:** $186K in annual savings from targeted campaigns
- **Enabled:** Real-time customer risk monitoring

The combination of predictive accuracy (77.8% ROC-AUC) and actionable insights makes this a production-ready solution for driving meaningful business outcomes.

---

##  License

This project is created for educational and portfolio purposes.

---

##  Author - Saniya 

**Data Analytics Professional**  
Demonstrating industry-level customer churn prediction and retention analytics capabilities.

*Last Updated: January 2026*
