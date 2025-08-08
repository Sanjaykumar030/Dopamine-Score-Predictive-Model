# Figures: Dopamine Score Predictive Model

This directory contains model evaluation and interpretability figures used in the research paper **"Modeling Dopamine-Triggering Patterns in Children's YouTube Videos Using Machine Learning."** The images illustrate classification performance and SHAP-based feature importance for each of the trained models.

---

## 📊 Confusion Matrices

Visualizing true positives, false positives, true negatives, and false negatives for each model. These matrices provide insight into overall model accuracy, sensitivity to dopamine-labeled content, and potential bias.

- `catboost_cmatrix.png` – Confusion matrix for the CatBoost model.
- `randomforest_cmatrix.png` – Confusion matrix for the Random Forest model.
- `xgboost_cmatrix.png` – Confusion matrix for the XGBoost model.

---

## 🧠 SHAP Summary Plots

SHAP (SHapley Additive exPlanations) visualizations for understanding feature contributions to individual predictions. Each plot shows the most impactful features ranked by average absolute SHAP value.

- `catboost_shap.png` – SHAP summary for the CatBoost model.
- `randomforest_shap.png` – SHAP summary for the Random Forest model.
- `xgboost_shap.png` – SHAP summary for the XGBoost model.

---

## 📁 Notes

- All SHAP values were computed using `shap.Explainer` from the SHAP Python library, with TreeExplainer for tree-based models.
- The figures were generated using Matplotlib and SHAP visualizations, exported at 300 DPI for publication quality.
- Class labels represent binary dopamine classification:  
  `1 = Dopamine-Inducing`  
  `0 = Not Dopamine-Inducing`

---
