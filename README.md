# Prédiction des Partis Politiques

Application d'analyse et de classification automatique de discours parlementaires européens (1999-2004) utilisant des techniques de traitement du langage naturel (NLP) et d'apprentissage automatique.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)

---

## Description

Ce projet vise à prédire l'affiliation politique de députés européens à partir de leurs discours parlementaires. Il utilise plusieurs algorithmes de classification supervisée pour identifier automatiquement le parti politique d'appartenance parmi cinq groupes principaux du Parlement européen.

### Objectifs

- Analyser des corpus multilingues de discours parlementaires (français, anglais, italien) [**Principal**]
- Comparer les performances de différents classifieurs de machine learning [**Principal**]
- Fournir une interface interactive pour l'analyse et la visualisation des résultats [Optionnel]
- Permettre la prédiction en temps réel sur de nouveaux textes [Optionnel]

---

## Fonctionnalités

### Script Principal (main.py)

- Pipeline complète d'analyse automatisée
- Génération de rapports détaillés (CSV, TXT)
- Création de graphiques de performance (PNG)
- Analyse comparative par langue et multilingue
- Export des résultats dans le dossier `resultats/`

### Application Streamlit Interactive

- **Page d'Accueil** : Vue d'ensemble du projet
- **Analyse & Entraînement** : 
  - Chargement des données par langue (FR, EN, IT) ou multilingue
  - Entraînement simultané de 6 modèles de classification
  - Visualisation des distributions de classes
  - Comparaison des performances en temps réel
- **Prédiction** :
  - Prédiction sur textes personnalisés
  - Test sur exemples aléatoires du corpus
  - Affichage des probabilités par classe
- **Visualisations** :
  - Comparaison des métriques des modèles
  - Matrices de confusion interactives
  - Performance détaillée par classe
- **À propos** : Documentation et informations sur le projet


---

## Structure du Projet

```
project_deft/
│
├── apprentissage/              # Données d'entraînement (XML)
│   ├── deft09_parlement_appr_fr.xml
│   ├── deft09_parlement_appr_en.xml
│   ├── deft09_parlement_appr_it.xml
│   └── deft09.dtd
│
├── test/                       # Données de test (XML)
│   ├── deft09_parlement_test_fr.xml
│   ├── deft09_parlement_test_en.xml
│   └── deft09_parlement_test_it.xml
│
├── reference/                  # Fichiers de référence (labels)
│   ├── deft09_parlement_ref_fr.txt
│   ├── deft09_parlement_ref_en.txt
│   └── deft09_parlement_ref_it.txt
│
├── resultats/                  # Résultats générés
│   ├── comparaison_modeles.csv
│   ├── rapport_detaille.txt
│   └── graphiques/
│
├── classifiers.py              # Définition des modèles ML
├── data_loader.py              # Chargement et parsing des données XML
├── evaluation_metrics.py       # Calcul des métriques de performance
├── visualizer.py               # Génération des graphiques
├── main.py                     # Script principal d'analyse
├── streamlit_app.py            # Application web interactive
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

---

## Installation

### Prérequis

- Python 3.13+

### Étapes d'installation

1. **Cloner le repository ou télécharger le projet**  



2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

Les packages suivants seront installés :
- `streamlit` : Framework pour l'interface web
- `pandas` : Manipulation de données
- `numpy` : Calculs numériques
- `matplotlib` : Visualisations
- `seaborn` : Graphiques statistiques
- `scikit-learn` : Algorithmes de machine learning

---

## Utilisation

### Option 1 : Script en Ligne de Commande

Exécutez le script principal pour une analyse complète :

```bash
python main.py
```

Les résultats seront générés dans le dossier `resultats/` :
- `comparaison_modeles.csv` : Tableau comparatif des performances
- `rapport_detaille.txt` : Rapport détaillé
- `graphiques/` : Tous les graphiques générés (PNG)


### Option 2 : Interface Web Interactive

Lancez l'application Streamlit :

```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : **http://localhost:8501**

#### Navigation dans l'application :

1. **Sélectionnez une langue** dans la barre latérale (FR, EN, IT ou Toutes)
2. Allez dans **"Analyse & Entraînement"**
3. Cliquez sur **"Charger les données"**
4. Cliquez sur **"Lancer l'entraînement"** pour entraîner les 6 modèles
5. Explorez les **visualisations** et testez les **prédictions** !


---

## Modèles de Classification

Le projet compare **6 classifieurs différents** :

| Modèle | Vectorisation | Caractéristiques |
|--------|---------------|------------------|
| **Naive Bayes (TF-IDF)** | TF-IDF | Rapide, baseline solide |
| **Naive Bayes (Count)** | CountVectorizer | Simple, efficace |
| **Complement Naive Bayes** | TF-IDF | Optimisé pour classes déséquilibrées |
| **Logistic Regression** | TF-IDF | Modèle linéaire robuste |
| **Linear SVM** | TF-IDF | Haute performance, bon généralisation |
| **Decision Tree** | TF-IDF | Interprétable, non-linéaire |

### Configuration des Vectoriseurs

- **max_features** : 3000-5000 (selon le modèle)
- **ngram_range** : (1, 2) - unigrammes et bigrammes
- **min_df** : 2 - fréquence minimale des termes
- **max_df** : 0.95 - filtrage des termes trop fréquents

---

## Métriques d'Évaluation

Le projet calcule plusieurs métriques pour chaque modèle :

- **Accuracy** : Taux de classification correcte global
- **Precision** : Précision des prédictions positives
- **Recall** : Taux de détection des vraies instances
- **F1-Score** : Moyenne harmonique de précision et recall
- **Métriques Macro/Micro** : Agrégations pour classes déséquilibrées
- **Matrice de confusion** : Visualisation des erreurs de classification
- **Performance par classe** : Analyse détaillée pour chaque parti

---

## Partis Politiques (Classes)

Le corpus contient des discours de **5 groupes politiques** du Parlement européen :

1. **PPE-DE** - Parti Populaire Européen - Démocrates Européens (centre-droit)
2. **PSE** - Parti Socialiste Européen (centre-gauche)
3. **ELDR** - Parti Européen des Libéraux, Démocrates et Réformateurs (centre)
4. **Verts/ALE** - Les Verts/Alliance Libre Européenne (écologistes)
5. **GUE-NGL** - Gauche Unitaire Européenne/Gauche Verte Nordique (gauche)

---

## Données

### Source
- **DEFT'09** (Défi Fouille de Textes 2009)
- Discours parlementaires européens (1999-2004)
- Corpus multilingue : français, anglais, italien

### Statistiques
- **~19,000 documents** d'entraînement
- **~13,000 documents** de test
- **~32,000 documents** au total
- **3 langues** supportées
- **5 classes** (partis politiques)

### Format
- Fichiers XML contenant :
  - ID du document
  - Langue
  - Parti politique (label)
  - Texte des discours

---

## Technologies Utilisées

### Langage
- **Python 3.13**

### Frameworks & Bibliothèques
- **Streamlit** : Interface web interactive
- **scikit-learn** : Algorithmes ML et métriques
- **pandas** : Manipulation et analyse de données
- **numpy** : Calculs numériques
- **matplotlib** : Visualisations statiques
- **seaborn** : Graphiques statistiques esthétiques

### Traitement du Langage Naturel
- **TfidfVectorizer** : Vectorisation TF-IDF
- **CountVectorizer** : Vectorisation par comptage
- **N-grammes** (unigrammes + bigrammes)

---

## Exemples de Résultats

### Comparaison des Modèles (exemple)

| Modèle | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Linear SVM | 0.8542 | 0.8520 | 0.8542 | 0.8515 |
| Logistic Regression | 0.8498 | 0.8475 | 0.8498 | 0.8472 |
| Naive Bayes (TF-IDF) | 0.8234 | 0.8210 | 0.8234 | 0.8205 |
| Complement NB | 0.8156 | 0.8132 | 0.8156 | 0.8128 |
| Naive Bayes (Count) | 0.8089 | 0.8065 | 0.8089 | 0.8060 |
| Decision Tree | 0.6723 | 0.6698 | 0.6723 | 0.6695 |

*Note : Les résultats réels peuvent varier selon le corpus et la langue.*

---

## Auteur

**Jean-Charles da Silva**
- GitHub: [@jcharlesDS](https://github.com/jcharlesDS)

---

## Licence

Ce projet utilise des données publiques du défi DEFT'09. Les données proviennent du Parlement européen et sont dans le domaine public.

Le code source est disponible pour usage académique et éducatif.

---

*Décembre 2025 - Janvier 2026*