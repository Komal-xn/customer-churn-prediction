# Customer Churn Prediction & Retention Analytics

End-to-end churn classification pipeline built on real telecom customer data (IBM Telco Customer Churn dataset). Covers data cleaning, EDA, feature engineering, model comparison, and K-Means customer segmentation.

## Dataset
- 454 real customer records, 21 raw features (demographics, account info, subscribed services)
- Churn rate: 24.9%
- Source: IBM Telco Customer Churn dataset (public)

## Pipeline
1. **Cleaning** — handled missing `TotalCharges` values for zero-tenure customers, type coercion
2. **EDA** — churn breakdown by contract type, tenure, monthly charges, internet service (see `eda_overview.png`)
3. **Feature Engineering** — 27 total features after encoding, including engineered `NumServices`, `AvgChargePerMonth`, `IsNewCustomer`, `HasMultipleServices`
4. **Modeling** — compared Logistic Regression, Random Forest, and Gradient Boosting
5. **Segmentation** — K-Means (k=4) clustering on tenure/charges/service-count to identify highest-risk customer segments

## Results

**Top 5 churn drivers** (by correlation with churn):
1. Contract type (month-to-month customers churn most)
2. Tenure (shorter tenure → higher churn risk)
3. Tech Support (no tech support → higher churn)
4. Online Security (no online security → higher churn)
5. Total Charges

**Model comparison:**

| Model | F1 Score | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.618 | 0.792 |
| Random Forest | 0.583 | 0.819 |
| Gradient Boosting | 0.578 | 0.808 |

Logistic Regression (with class balancing) gave the best F1 score on this dataset — recall of 74% on churners, meaning it catches most at-risk customers, which matters more than raw accuracy for a retention use case.

**Customer segments (K-Means, k=4):**

| Cluster | Customers | Churn Rate | Avg Tenure | Avg Monthly Charge | Avg Services |
|---|---|---|---|---|---|
| 0 | 67 | 4.0% | 54.1 mo | $30.59 | 0.8 |
| 1 | 128 | **38.0%** | 7.3 mo | $42.47 | 0.6 |
| 2 | 108 | 13.0% | 60.7 mo | $92.55 | 5.1 |
| 3 | 151 | 31.0% | 22.3 mo | $82.19 | 3.0 |

**Cluster 1 is the highest-risk segment** — 38% churn rate, short tenure (~7 months), low service adoption. This group (28% of the customer base) is the clearest retention-outreach priority: new customers who haven't adopted enough services to become sticky.

## Files
- `churn_pipeline.py` — full pipeline, run end-to-end
- `telco_churn_raw.csv` — raw dataset
- `eda_overview.png`, `confusion_matrix.png`, `feature_importance.png`, `customer_segments.png` — visualizations
- `results_summary.txt` — plain-text results output

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python churn_pipeline.py
```
