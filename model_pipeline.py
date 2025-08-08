import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Core Machine Learning Libraries
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Scikit-learn Tools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef, log_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Optimization and Model Interpretability
import optuna
import shap

# General Configuration
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Step 1: Load and prepare the dataset
df = pd.read_csv("Dopamine_Data.csv")
df_clean = df.dropna(subset=["dopamine_label"]).copy()

# Feature: Log-transformed view count
if "view_count" in df_clean.columns:
    df_clean["log_view_count"] = np.log1p(df_clean["view_count"])

# Feature: Extract date components
if "date_published" in df_clean.columns:
    df_clean['date_published'] = pd.to_datetime(df_clean['date_published'], format='%d-%m-%Y', errors='coerce')
    df_clean["publish_year"] = df_clean["date_published"].dt.year
    df_clean["publish_month"] = df_clean["date_published"].dt.month
    df_clean["publish_dayofweek"] = df_clean["date_published"].dt.dayofweek
    df_clean["is_weekend"] = df_clean["publish_dayofweek"].isin([5, 6]).astype(int)

# Feature selection
cols_to_drop = ["dopamine_label", "video_id", "channel_name", "video_title", "date_published",  "view_count"]
X = df_clean.drop(columns=cols_to_drop, errors="ignore")
y = df_clean["dopamine_label"]

# Data preprocessing
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

for col in categorical_features:
    X[col] = X[col].fillna("missing_value").astype(str)
for col in numerical_features:
    X[col] = X[col].fillna(X[col].median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 2: Define preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
], remainder="passthrough")

# Step 2.5: Hyperparameter tuning using Optuna
N_TRIALS = 30
CV_SPLITS = 5
cv_strategy = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'cat_features': categorical_features,
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1).mean()

def objective_xgboost(trial):
    params = {
        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 200, 1000),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.1, log=True),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 4, 8),
        'classifier__reg_lambda': trial.suggest_float('classifier__reg_lambda', 1, 10, log=True),
    }
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss',))
    ])
    pipeline.set_params(**params)
    return cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1).mean()

def objective_rf(trial):
    params = {
        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 100, 800),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 5, 25),
        'classifier__min_samples_split': trial.suggest_int('classifier__min_samples_split', 2, 12),
        'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 6),
    }
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    pipeline.set_params(**params)
    return cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1).mean()

# Optimization Execution
objectives = {"CatBoost": objective_catboost, "XGBoost": objective_xgboost, "RandomForest": objective_rf}
tuned_models = {}

for model_name, objective_func in objectives.items():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=N_TRIALS)
    best_params = study.best_params
    if model_name == "CatBoost":
        model = CatBoostClassifier(**best_params, cat_features=categorical_features, random_seed=42, verbose=0)
    else:
        base_model = XGBClassifier if model_name == "XGBoost" else RandomForestClassifier
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", base_model(random_state=42))
        ])
        if model_name == "XGBoost":
            best_params.update({"classifier__eval_metric": "logloss", "classifier__use_label_encoder": False})
        else:
            best_params["classifier__n_jobs"] = -1
        model.set_params(**best_params)
    tuned_models[model_name] = model

models = tuned_models

# Step 3-4: Cross-validation results aggregation
results_df = pd.DataFrame(columns=['Model', 'Fold', 'Accuracy', 'ROC_AUC', 'F1-Score', 'Precision', 'Recall', 'MCC', 'LogLoss'])

for model_name, model in models.items():
    for fold_num, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        metrics = {
            'Model': model_name,
            'Fold': fold_num,
            'Accuracy': accuracy_score(y_fold_val, y_pred),
            'ROC_AUC': roc_auc_score(y_fold_val, y_proba),
            'F1-Score': f1_score(y_fold_val, y_pred),
            'Precision': precision_score(y_fold_val, y_pred),
            'Recall': recall_score(y_fold_val, y_pred),
            'MCC': matthews_corrcoef(y_fold_val, y_pred),
            'LogLoss': log_loss(y_fold_val, y_proba)
        }
        results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)

cv_summary = results_df.groupby('Model').agg(Avg_Accuracy=('Accuracy', 'mean'), Avg_ROC_AUC=('ROC_AUC', 'mean'), Avg_F1_Score=('F1-Score', 'mean'), Avg_Precision=('Precision', 'mean'), Avg_Recall=('Recall', 'mean'), Avg_MCC=('MCC', 'mean'), Std_ROC_AUC=('ROC_AUC', 'std')).reset_index().sort_values(by='Avg_ROC_AUC', ascending=False)
print(cv_summary.to_string())

# Step 5: Final model evaluation and feature interpretation
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_final = model.predict(X_test)
    y_pred_proba_final = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {model_name} Test Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}, ROC AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
    print(classification_report(y_test, y_pred_final))

    cm = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if model_name == "CatBoost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        shap.summary_plot(shap_values, X_test)
    else:
        classifier = model.named_steps["classifier"]
        X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_test_df)

        if model_name == "RandomForest":
            importances = classifier.feature_importances_
            pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20).plot(kind="bar", figsize=(12, 6))
            plt.title("Random Forest Feature Importance")
            plt.tight_layout()
            plt.show()

            shap.plots.beeswarm(shap_values[:, :, 1], max_display=15)
        else:
            shap.summary_plot(shap_values, X_test_df)

# Step 6: Learning Curve Visualization
def plot_learning_curve(estimator, X, y, title):
    sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv_strategy, scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    plt.plot(sizes, train_scores.mean(axis=1), label='Train')
    plt.plot(sizes, val_scores.mean(axis=1), label='Validation')
    plt.fill_between(sizes, train_scores.mean(axis=1) - train_scores.std(axis=1), train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(sizes, val_scores.mean(axis=1) - val_scores.std(axis=1), val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.title(f"Learning Curve - {title}")
    plt.xlabel("Training Samples")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for model_name, model in models.items():
    plot_learning_curve(model, X_train, y_train, model_name)
