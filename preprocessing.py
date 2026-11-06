import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_data(path="processed.cleveland.data"):
    if not os.path.exists(path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        df = pd.read_csv(url, names=COLUMNS, na_values=["?", "NA"])
        df.to_csv(path, index=False, header=False)
    else:
        df = pd.read_csv(path, names=COLUMNS, na_values=["?", "NA"])

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df.drop(columns=["target"])
    y = (df["target"] > 0).astype(int).values
    return X, y


def get_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipeline, NUM_FEATURES),
        ("cat", cat_pipeline, CAT_FEATURES)
    ])


def run_pca(X_processed, threshold=0.90):
    pca = PCA()
    pca.fit(X_processed)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.argmax(cum_var >= threshold) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Seuil {int(threshold*100)}%")
    plt.axvline(x=n_comp, color='g', linestyle=':', label=f"{n_comp} composantes")
    plt.title("Critère d'Arpenteur : Variance Expliquée par l'ACP")
    plt.xlabel("Nombre de Composantes")
    plt.ylabel("Variance Cumulée")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_scree_plot.png")
    plt.close()

    return n_comp, cum_var[n_comp - 1]

