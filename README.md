# Projet 3 : Support au Diagnostic Médical (Cleveland Heart Disease)
> **Module :** IA Avancées — ING GLSI / AI-DS  
> **Enseignant :** Dr. Azer Hasnaoui  
> **Auteur :** Projet Étudiant — Année Universitaire 2026-2027

---

## 📌 Présentation du Projet

Ce projet implémente un système d'aide à la décision médicale pour le diagnostic des **maladies coronariennes (Coronary Artery Disease - CAD)** à partir du jeu de données de référence de la clinique de Cleveland (*UCI Machine Learning Repository*).

L'objectif est d'appliquer rigoureusement la méthodologie d'apprentissage supervisé et non supervisé vue en cours :
1. **Prétraitement & Typologie des données manquantes (Séance 7)** : Imputation multivariée KNN, encodage One-Hot avec suppression de la modalité de référence (`drop='first'`) pour éviter le *Dummy Variable Trap*, et standardisation.
2. **Réduction de dimension (Séance 7)** : Analyse en Composantes Principales (ACP) avec tracé du critère d'arpenteur (*Scree Plot*) pour retenir $\ge 90\%$ de variance.
3. **Modélisation d'ensemble & Deep Learning (Séances 2 & 6)** : Comparaison entre **Random Forest** (réduction de variance), **XGBoost** (réduction de biais avec pénalités $L_1/L_2$) et un **Perceptron Multicouche (DeepMLP)** développé sous **PyTorch**.
4. **Validation croisée & Évaluation médicale (Séance 8)** : Évaluation sans fuite de données (*data leakage free*) via une validation croisée stratifiée ($K=5$). Priorité accordée au **Rappel (Sensibilité)** afin de minimiser le risque critique de **Faux Négatifs** en milieu hospitalier.

---

## 🩺 Description des Données (Cleveland Dataset)

Le jeu de données comprend **303 patients** et **14 attributs cliniques** (13 variables explicatives et 1 variable cible) :

| Variable | Type | Description médicale | Valeurs / Unités |
| :--- | :--- | :--- | :--- |
| `age` | Continu | Âge du patient | Années |
| `sex` | Binaire | Sexe biologique | $1 = \text{Homme}, 0 = \text{Femme}$ |
| `cp` | Catégoriel | Type de douleur thoracique | 1: Angor typique, 2: Angor atypique, 3: Douleur non angineuse, 4: Asymptomatique |
| `trestbps` | Continu | Pression artérielle au repos à l'admission | mm Hg |
| `chol` | Continu | Cholestérol sérique | mg/dl |
| `fbs` | Binaire | Glycémie à jeun > 120 mg/dl | $1 = \text{Vrai}, 0 = \text{Faux}$ |
| `restecg` | Catégoriel | Résultats électrocardiographiques au repos | 0: Normal, 1: Anomalie onde ST-T, 2: Hypertrophie ventriculaire gauche |
| `thalach` | Continu | Fréquence cardiaque maximale atteinte à l'effort | Battements par minute (bpm) |
| `exang` | Binaire | Angine de poitrine induite par l'effort | $1 = \text{Oui}, 0 = \text{Non}$ |
| `oldpeak` | Continu | Dépression du segment ST induite par l'effort relative au repos | mm (mesure d'ischémie myocardique) |
| `slope` | Catégoriel | Pente du segment ST à l'effort maximal | 1: Pente ascendante, 2: Plat, 3: Pente descendante |
| `ca` | Discret | Nombre de vaisseaux majeurs (0-3) colorés par fluoroscopie | 0 à 3 (*4 valeurs manquantes MAR*) |
| `thal` | Catégoriel | Scintigraphie myocardique au Thallium-201 | 3: Normal, 6: Défaut fixe (infarctus), 7: Défaut réversible (*2 valeurs manquantes MAR*) |
| `target` | Cible | Diagnostic angiographique de sténose coronarienne | $0 = \text{Sain}$ ($< 50\%$ rétrécissement) / $1 = \text{Malade}$ ($\ge 50\%$ rétrécissement) |

---

## 🛠️ Installation & Préparation de l'Environnement

Suivez ces étapes simples pour cloner et exécuter le projet sur votre machine (conforme à la **Séance 1 Slide 16**) :

### 1. Cloner le dépôt et ouvrir le dossier
```bash
git clone <URL_DU_DEPOT_GITHUB>
cd "AI Project"
```

### 2. Créer l'environnement virtuel Python
```bash
python -m venv .venv
```

### 3. Activer l'environnement virtuel
- **Sur Windows (PowerShell) :**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Sur Windows (CMD) :**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Sur macOS / Linux :**
  ```bash
  source .venv/bin/activate
  ```

### 4. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 📁 Architecture Modulaire du Code

Le projet est découpé en modules autonomes et concis :

```
AI Project/
│
├── AI/                           # Diapositives du cours (Séances 1 à 8)
├── processed.cleveland.data      # Cache local du dataset (303 lignes)
├── preprocessing.py              # Séance 7 : Imputation KNN, One-Hot drop='first', StandardScaler, ACP
├── models.py                     # Séances 2 & 6 : Random Forest, XGBoost, DeepMLP PyTorch
├── evaluate.py                   # Séance 8 : Stratified K-Fold (K=5), métriques, Boxplot, Matrices
├── main.py                       # Script principal orchestrant l'exécution
├── projet_diagnostic.ipynb       # Notebook Jupyter pas-à-pas pour rendu interactif
├── requirements.txt              # Liste des bibliothèques requises
└── README.md                     # Documentation complète pour GitHub
```

---

## 🚀 Utilisation & Exécution

### Option A : Exécution via le script terminal
```bash
python main.py
```

### Option B : Exécution interactive via Jupyter Notebook
Lancez Jupyter Notebook ou ouvrez directement [`projet_diagnostic.ipynb`](projet_diagnostic.ipynb) sous VS Code :
```bash
jupyter notebook projet_diagnostic.ipynb
```

---

## 📊 Résultats & Analyse Comparative (5-Fold Stratified CV)

L'évaluation sans fuite d'information (*leak-free*) sur les 5 plis donne les résultats suivants :

| Modèle | Rappel (Sensibilité) ($\uparrow$) | Précision | Score F1 | ROC-AUC | Alignement avec le cours |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | $0.755 \pm 0.036$ | $\mathbf{0.818 \pm 0.054}$ | $0.784 \pm 0.025$ | $\mathbf{0.904 \pm 0.025}$ | **Séance 2** : Réduction de variance par Bagging |
| **XGBoost** | $\mathbf{0.776 \pm 0.055}$ | $0.814 \pm 0.065$ | $\mathbf{0.791 \pm 0.025}$ | $0.899 \pm 0.020$ | **Séance 2** : Réduction de biais avec pénalités $L_1/L_2$ |
| **PyTorch DeepMLP** | $0.733 \pm 0.068$ | $0.822 \pm 0.055$ | $0.771 \pm 0.033$ | $0.865 \pm 0.013$ | **Séance 6** : Réseau séquentiel `nn.Sequential` (ReLU, Dropout, Adam) |

---

## 📈 Graphiques Générés

L'exécution produit automatiquement 4 visualisations directement issues des diapositives :

1. **`pca_scree_plot.png` (Séance 7 Slide 15)** :  
   Critère d'Arpenteur montrant que **11 composantes** suffisent pour capturer **91.5%** de la variance totale.
2. **`models_stability_boxplot.png` (Séance 8 Slide 17)** :  
   Diagramme en boîte comparant la dispersion et la stabilité du Rappel à travers les plis de validation croisée.
3. **`confusion_matrix_and_roc.png` (Séance 8 Slide 12)** :  
   Matrice de confusion pour observer les Faux Négatifs / Faux Positifs et courbes ROC comparées.
4. **`feature_importance.png` (Séance 2)** :  
   Importance relative des prédicteurs calculée par XGBoost.

---

## 💡 Justifications Théoriques du Cours

### 1. Pourquoi le Rappel (Sensibilité) prime en diagnostic médical ? (Séance 8 Slide 6)
Dans le secteur de la santé, le coût des erreurs est asymétrique :
$$C(\text{Faux Négatif}) \gg C(\text{Faux Positif})$$
- **Faux Négatif (FN)** : Un patient atteint d'une sténose critique est déclaré sain et renvoyé chez lui sans surveillance, risquant un infarctus du myocarde mortel.
- **Faux Positif (FP)** : Un patient sain est orienté vers des examens cardiologiques complémentaires (coronarographie, échocardiographie d'effort). La conséquence est financière et émotionnelle, mais n'engage pas le pronostic vital.

### 2. Pourquoi `drop='first'` dans OneHotEncoder ? (Séance 7 Slides 8–10)
Pour une variable à $K$ modalités, générer $K$ colonnes binaires crée une dépendance linéaire exacte :
$$\sum_{k=1}^K x_{i,k} = 1 = x_{i, \text{intercept}} \quad \forall i$$
C'est le **Dummy Variable Trap** : la matrice $\mathbf{X}^T\mathbf{X}$ devient singulière ($\det(\mathbf{X}^T\mathbf{X}) = 0$) et non-inversible. L'option `drop='first'` supprime la modalité de référence pour garantir l'indépendance linéaire des coefficients.

### 3. Principaux biomarqueurs identifiés
- **`thal` (Thallium)** : Les défauts réversibles identifient les territoires myocardiques viables mais ischémiques sous stress.
- **`ca` (Fluoroscopie)** : Quantifie directement le fardeau athéroscléreux calcifié dans les artères coronaires majeures.
- **`cp` (Angor asymptomatique)** : Très représentatif de l'ischémie silencieuse, fréquente chez les patients diabétiques.
