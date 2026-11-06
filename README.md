# Projet 3 : Support au Diagnostic Médical – Heart Disease

**Module :** IA Avancées – ING GLSI / AI-DS  
**Enseignant :** Dr. Azer Hasnaoui  
**Année universitaire :** 2026–2027

## 1. Introduction

Ce projet consiste à utiliser des techniques de Machine Learning pour prédire la présence d'une maladie cardiaque à partir de données médicales.

Nous avons testé trois modèles : Random Forest, XGBoost et un réseau de neurones (DeepMLP avec PyTorch). Le but est de comparer leurs résultats et de voir lequel donne les meilleures performances.

Le travail comprend le nettoyage des données, le traitement des valeurs manquantes, l'encodage des variables, la standardisation, la réduction de dimension avec PCA et enfin l'entraînement et l'évaluation des modèles.

## 2. Dataset

Nous avons utilisé le **Cleveland Heart Disease Dataset**, disponible sur le [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

Il contient 303 observations et 14 attributs, dont 13 caractéristiques médicales et une variable cible.

Les données concernent notamment :
- L'âge et le sexe du patient.
- Le type de douleur thoracique.
- La pression artérielle et le taux de cholestérol.
- La fréquence cardiaque maximale.
- Les résultats de certains examens cardiaques.

La variable cible comporte deux classes :
- `0` : absence de maladie cardiaque.
- `1` : présence de maladie cardiaque.

## 3. Technologies utilisées

- **Python** : développement du projet.
- **Pandas et NumPy** : traitement des données.
- **Scikit-learn** : prétraitement, PCA et évaluation.
- **XGBoost** : modèle de classification.
- **PyTorch** : réseau de neurones.
- **Matplotlib et Seaborn** : visualisation des résultats.
- **Jupyter Notebook** : expérimentation et présentation du travail.

## 4. Installation

### Étape 1 : Cloner le projet

```bash
git clone https://github.com/louatiaminn/heart-disease-diagnostic-ai.git
cd heart-disease-diagnostic-ai
```

### Étape 2 : Créer un environnement virtuel

```bash
python -m venv .venv
```

### Étape 3 : Activer l'environnement

**Windows :**
```bash
.venv\Scripts\activate
```

**Linux / macOS :**
```bash
source .venv/bin/activate
```

### Étape 4 : Installer les bibliothèques

```bash
pip install -r requirements.txt
```

## 5. Exécution

Pour lancer le projet depuis le terminal :

```bash
python main.py
```

Il est également possible d'ouvrir `projet_diagnostic.ipynb` avec Jupyter Notebook ou VS Code pour consulter les différentes étapes.

## 6. Résultats

Les modèles ont été comparés avec une validation croisée stratifiée à 5 plis (*Stratified K-Fold*).

| Modèle | Recall | Precision | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.755 | 0.818 | 0.784 | 0.904 |
| XGBoost | 0.776 | 0.814 | 0.791 | 0.899 |
| DeepMLP | 0.733 | 0.822 | 0.771 | 0.865 |

XGBoost obtient le meilleur Recall et le meilleur F1-Score, tandis que Random Forest obtient le meilleur ROC-AUC.

## 7. Équipe du projet

- Mohamed Amine Loueti
- Jesser Marzouki
- Souhail Omar

---

*Projet réalisé dans le cadre du module IA Avancées – 2026/2027.*
