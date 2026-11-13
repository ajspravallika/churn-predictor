# Customer Churn Predictor

Predicts which telecom subscribers are likely to cancel, so a company can contact them before they leave.

**Live demo:** https://churnwatch-ml-dashboard.streamlit.app
(The free host puts the app to sleep when nobody visits, so the first load may take a few seconds.)

![High-risk customer](screenshots/high-risk.png)
![Low-risk customer](screenshots/low-risk.png)

## Problem

When a subscriber cancels (churns), the company loses their future revenue, and winning a new customer usually costs more than keeping an existing one. This project predicts each customer's chance of leaving, so a retention team can contact the riskiest customers first.

## Data

- [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle: 7,043 customers and 21 columns (an ID, 19 input columns, and the churn label)
- 26.5% of customers churned, so the classes are imbalanced
- Cleaning: `TotalCharges` was stored as text, and 11 blank values belong to brand-new customers (tenure 0), so they were filled with 0. `customerID` was dropped because it carries no information

## What the data showed

| Finding | Result |
|---|---|
| Month-to-month contracts | 42.7% churned (one year: 11.3%, two year: 2.8%) |
| First year of tenure | 47.4% churned, falling to 9.5% for customers with 49 to 72 months |
| Churners paid more per month | About 74 vs 61 on average |

## Approach

1. Explored the data and wrote down churn patterns
2. Built a preprocessing pipeline (scaling for numbers, one-hot encoding for categories)
3. Trained two models on an 80/20 train/test split: Logistic Regression and Random Forest, both with `class_weight="balanced"`
4. Compared them on ROC-AUC, recall, and precision
5. Tuned the decision cutoff for the business goal (catching churners)
6. Saved the final pipeline and deployed a Streamlit dashboard

## Results (test set of 1,409 customers)

| Model | ROC-AUC | Recall (churners) at 0.4 | Precision at 0.4 |
|---|---|---|---|
| Logistic Regression | 0.842 | 0.866 | 0.467 |
| Random Forest | 0.842 | 0.816 | 0.510 |

**Chosen model:** Random Forest with a cutoff of 0.4. It catches about 82% of customers who leave, and about half of the customers it flags really are at risk. Missing a churner costs more than sending an extra offer, so recall was prioritised over precision.

Both models reach the same ROC-AUC (0.842), so the simpler model matched the more complex one on this dataset.

## What the model relies on

Permutation importance on the test set (shuffle one column and measure the drop in ROC-AUC):

![Feature importance](screenshots/chart.png)

Contract type is the strongest signal, followed by tenure and internet service. Gender does not appear in the top 10, and demographic columns such as SeniorCitizen contribute very little. This shows what the model uses, not why customers leave.

## Run locally

```bash
git clone https://github.com/ajspravallika/churn-predictor.git
cd churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

The saved model was trained with scikit-learn 1.6.1 on Python 3.13, so use the same versions.

## Project structure

```
app.py              Streamlit dashboard
churn_model.pkl     Trained Random Forest pipeline
cutoff.txt          Decision cutoff (0.4)
importance.csv      Permutation importance scores shown in the dashboard
notebooks/          Exploration, training, and evaluation (Colab notebook)
screenshots/        Images used in this README
requirements.txt    Dependencies
```

## Limitations

- The dataset is a single snapshot from a telecom company, not live subscription data
- The cutoff was chosen by looking at the same test set it was scored on. A separate validation set would be stricter
- `TotalCharges` in the dashboard is estimated as tenure x monthly charges
- The model predicts a probability and cannot know what an individual customer is thinking
- Feature importance shows what the model relies on, not the cause of churn

## Future work

- Explain individual predictions with SHAP
- Cross-validation and a separate validation set for cutoff selection
- Serve the model through an API (FastAPI) with a separate frontend

## Tech stack

Python, pandas, scikit-learn, Streamlit, Altair, joblibgit add .