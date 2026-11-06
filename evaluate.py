import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from preprocessing import get_preprocessor
from models import get_random_forest, get_xgboost, train_pytorch, predict_pytorch

SEED = 42


def evaluate_models(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models_scores = {
        "Random Forest": {"accuracy": [], "recall": [], "precision": [], "f1": [], "roc_auc": []},
        "XGBoost": {"accuracy": [], "recall": [], "precision": [], "f1": [], "roc_auc": []},
        "PyTorch MLP": {"accuracy": [], "recall": [], "precision": [], "f1": [], "roc_auc": []},
    }

    all_y_test = []
    y_probs_dict = {"Random Forest": [], "XGBoost": [], "PyTorch MLP": []}
    y_preds_dict = {"Random Forest": [], "XGBoost": [], "PyTorch MLP": []}

    for train_idx, test_idx in cv.split(X, y):
        X_tr_raw, X_te_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        all_y_test.extend(y_test)

        prep = get_preprocessor()
        X_train = prep.fit_transform(X_tr_raw)
        X_test = prep.transform(X_te_raw)

        # 1. Random Forest
        rf = get_random_forest()
        rf.fit(X_train, y_train)
        rf_p = rf.predict_proba(X_test)[:, 1]
        rf_pred = rf.predict(X_test)
        _record_metrics(models_scores["Random Forest"], y_test, rf_pred, rf_p)
        y_probs_dict["Random Forest"].extend(rf_p)
        y_preds_dict["Random Forest"].extend(rf_pred)

        # 2. XGBoost
        xgb_clf = get_xgboost()
        xgb_clf.fit(X_train, y_train)
        xgb_p = xgb_clf.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_clf.predict(X_test)
        _record_metrics(models_scores["XGBoost"], y_test, xgb_pred, xgb_p)
        y_probs_dict["XGBoost"].extend(xgb_p)
        y_preds_dict["XGBoost"].extend(xgb_pred)

        # 3. PyTorch DeepMLP
        mlp = train_pytorch(X_train, y_train, epochs=80, lr=0.01)
        mlp_pred, mlp_p = predict_pytorch(mlp, X_test)
        _record_metrics(models_scores["PyTorch MLP"], y_test, mlp_pred, mlp_p)
        y_probs_dict["PyTorch MLP"].extend(mlp_p)
        y_preds_dict["PyTorch MLP"].extend(mlp_pred)

    return models_scores, all_y_test, y_probs_dict, y_preds_dict


def _record_metrics(store, y_true, y_pred, y_prob):
    store["accuracy"].append(accuracy_score(y_true, y_pred))
    store["recall"].append(recall_score(y_true, y_pred))
    store["precision"].append(precision_score(y_true, y_pred))
    store["f1"].append(f1_score(y_true, y_pred))
    store["roc_auc"].append(roc_auc_score(y_true, y_prob))


def plot_evaluation(models_scores, all_y_test, y_probs_dict, y_preds_dict):
    # 1. Boxplot de stabilité du Rappel
    plt.figure(figsize=(7, 4))
    plt.boxplot([models_scores[m]["recall"] for m in models_scores], tick_labels=list(models_scores.keys()))
    plt.title("Stabilité du Rappel (Sensibilité) par Validation Croisée")
    plt.ylabel("Rappel")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("models_stability_boxplot.png")
    plt.close()

    # 2. Matrice de Confusion et ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    cm = confusion_matrix(all_y_test, y_preds_dict["XGBoost"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Sain", "Malade"], yticklabels=["Sain", "Malade"])
    ax1.set_title("Matrice de Confusion : XGBoost")
    ax1.set_xlabel("Prédit")
    ax1.set_ylabel("Réel")

    for name in models_scores:
        fpr, tpr, _ = roc_curve(all_y_test, y_probs_dict[name])
        auc_val = roc_auc_score(all_y_test, y_probs_dict[name])
        ax2.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax2.set_title("Courbes ROC")
    ax2.set_xlabel("Taux de Faux Positifs")
    ax2.set_ylabel("Taux de Vrais Positifs (Rappel)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("confusion_matrix_and_roc.png")
    plt.close()


def plot_feature_importance(xgb_model, feature_names):
    importances = xgb_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:6]

    plt.figure(figsize=(7, 3.5))
    plt.barh([feature_names[i] for i in reversed(top_idx)], [importances[i] for i in reversed(top_idx)], color="#1f77b4")
    plt.title("Importance des Variables (XGBoost)")
    plt.xlabel("Importance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

