import numpy as np
import matplotlib
matplotlib.use("Agg")

from preprocessing import load_data, get_preprocessor, run_pca
from models import get_xgboost
from evaluate import evaluate_models, plot_evaluation, plot_feature_importance


def main():
    # 1. Chargement des données
    X, y = load_data()
    print(f"Dataset : {X.shape[0]} lignes, {X.shape[1]} variables.")
    print(f"Classes : {np.sum(y == 0)} Sains, {np.sum(y == 1)} Malades.")

    # 2. Prétraitement et ACP
    prep = get_preprocessor()
    X_proc = prep.fit_transform(X)
    raw_names = prep.get_feature_names_out()
    feature_names = [f.split("__")[-1] for f in raw_names]

    n_comp, var_ratio = run_pca(X_proc, threshold=0.90)
    print(f"ACP : {n_comp} composantes expliquent {var_ratio*100:.1f}% de la variance.")

    # 3. Validation croisée stratifiée (K=5)
    print("\nÉvaluation des modèles en validation croisée (5 plis)...")
    scores, y_true, y_probs, y_preds = evaluate_models(X, y)

    # 4. Affichage des résultats
    print("\n--- RÉSULTATS COMPARATIFS (Moyenne +/- Écart-Type) ---")
    print(f"{'Modèle':<15} | {'Rappel':<18} | {'Précision':<18} | {'Score F1':<18} | {'ROC-AUC':<18}")
    print("-" * 75)
    for model_name, metrics in scores.items():
        rec = f"{np.mean(metrics['recall']):.3f} +/- {np.std(metrics['recall']):.3f}"
        prec = f"{np.mean(metrics['precision']):.3f} +/- {np.std(metrics['precision']):.3f}"
        f1 = f"{np.mean(metrics['f1']):.3f} +/- {np.std(metrics['f1']):.3f}"
        auc = f"{np.mean(metrics['roc_auc']):.3f} +/- {np.std(metrics['roc_auc']):.3f}"
        print(f"{model_name:<15} | {rec:<18} | {prec:<18} | {f1:<18} | {auc:<18}")

    # 5. Graphiques d'évaluation et importance des variables
    plot_evaluation(scores, y_true, y_probs, y_preds)

    xgb_model = get_xgboost()
    xgb_model.fit(X_proc, y)
    plot_feature_importance(xgb_model, feature_names)

    print("\nGraphiques sauvegardés :")
    print("- pca_scree_plot.png")
    print("- models_stability_boxplot.png")
    print("- confusion_matrix_and_roc.png")
    print("- feature_importance.png")


if __name__ == "__main__":
    main()

