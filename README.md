# LoanTap Credit-Risk Pipeline

An end-to-end proof-of-concept for predicting loan defaults using Logistic Regression, SMOTE, and class-weighting.

## 🚀 Project Overview
- Built a binary classifier to flag “Charged Off” loans vs “Fully Paid.”  
- 370K+ cleaned records, 22 engineered features (borrower profile, credit history, loan terms).  
- Tackled severe class imbalance (80:20) with SMOTE & custom weights.  
- Tuned for **F1-score** on defaulters (minority class) rather than raw accuracy.

## 🛠️ Tech Stack
Python • pandas • scikit-learn • imbalanced-learn • category_encoders • statsmodels • seaborn • matplotlib

## 📊 Key Steps
1. **Cleaning & Imputation**  
   - Mode-fill and binarize `mort_acc`  
   - Drop high-cardinality/text cols (`emp_title`, `address`, etc.)  
2. **Feature Engineering**  
   - Label & target encoding (`term`, `grade`, `purpose`)  
   - Extract `issue_month`, `issue_year`, `credit_age_years`  
3. **Collinearity Control**  
   - VIF analysis → remove `loan_amnt` (collinear with `installment`)  
4. **Modeling & Tuning**  
   - Baseline logistic regression → F1(defaulter)=0.13  
   - +Class weights & SMOTE → F1=0.42  
   - Threshold sweep & PR-AUC evaluation  
5. **Evaluation**  
   - Precision-recall curves, confusion matrices, test-set validation  

## 📈 Results
- **Defaulter F1**: 0.13 → **0.42** (+223%)  
- **Recall**: 0.08 → 0.64 (majority of defaulters caught)  
- **Precision**: 0.53 → 0.31 (trade-off for higher recall)  
- **PR-AUC**: ~0.36 before & after balancing (reflects minority performance)  

## 🔍 Business Insights & Conclusions
- **High-risk segments**:  
  - 60-month terms, grades E–G, high DTI, high interest  
  - Surprisingly, higher open-account counts correlate with default  
- **Model learnings**:  
  - Grade is the strongest predictor  
  - Multicollinearity (installment vs loan_amnt) didn’t harm F1  
- **Metric choice**:  
  - F1-score & recall matter most for NPA control  
  - AUROC alone is misleading under imbalance  

## 🎯 Actionable Recommendations
1. **Refine underwriting rules**  
   - Stricter approval or higher rates for 60-month loans  
   - Tighter DTI and loan-amount caps on E–G grades  
2. **Monitor key flags**  
   - Any public record → automatic manual review  
   - Small-business purpose → require extra collateral  
3. **Operationalize model**  
   - Deploy with a 0.5 threshold for balanced F1  
   - Regularly retrain as economic conditions shift  
4. **Enhance data sources**  
   - Add transaction histories, geo-demographics, alternative credit data  
5. **Cost-benefit analysis**  
   - Quantify cost of false positives vs false negatives to set business threshold  

---

> **Author:** Rahil Qureshi |  Data Analyst/Scientist  
> Feel free to ⭐ this repo or connect if you're hiring in data science & analytics!  
