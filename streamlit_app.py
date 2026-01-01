"""
Application Streamlit pour l'analyse de prédiction de partis politiques.
Interface interactive pour explorer les données, comparer les classifieurs
et visualiser les résultats.
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_loader import DataLoader
from classifiers import ClassifierFactory
from evaluation_metrics import EvaluationMetrics

# Initialiser NLTK au démarrage de l'application
@st.cache_resource
def setup_nltk():
    """Configure NLTK et télécharge les stopwords nécessaires"""
    try:
        import nltk
        import ssl
        
        # Contourner les problèmes SSL
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Télécharger les stopwords si nécessaire
        try:
            from nltk.corpus import stopwords
            stopwords.words('french')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        return True
    except Exception as e:
        st.warning(f"Impossible de configurer NLTK: {e}")
        return False

# Configurer NLTK au démarrage
nltk_ready = setup_nltk()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction - Partis Politiques",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}  # Dictionnaire pour stocker les modèles entraînés
if 'results' not in st.session_state:
    st.session_state.results = {}  # Dictionnaire pour stocker les résultats d'évaluation
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False  # Indicateur de chargement des données
if 'loaded_language' not in st.session_state:
    st.session_state.loaded_language = None  # Langue des données actuellement chargées
if 'page' not in st.session_state:
    st.session_state.page = "Accueil"  # Page par défaut

# Titre principal
st.markdown('<h1 class="main-header"> Prédiction des Partis Politiques</h1>', unsafe_allow_html=True)
st.markdown("### Analyse interactive des textes parlementaires européens. (1999-2004)")
st.divider()

# Sidebar
with st.sidebar:
    st.header("Configurations")
    
    # Sélection de la page
    page = st.selectbox(
        "Choisir une page:",
        ["Accueil", "Analyse & Entraînement", "Prédiction", "Visualisations", "À propos"],
        index=["Accueil", "Analyse & Entraînement", "Prédiction", "Visualisations", "À propos"].index(st.session_state.page)
    )
    
    # Mettre à jour le session_state si la sélection change
    if page != st.session_state.page:
        st.session_state.page = page
    
    st.divider()
    
    # Sélection de la langue
    st.subheader("Langue")
    language = st.selectbox(
        "Choisir la langue des données:",
        ["fr", "en", "it", "all"],
        format_func=lambda x: {"fr": "Français", "en": "Anglais",
                    "it": "Italien", "all": "Toutes"}[x]
    )
    
    st.divider()
    
    # Informations
    st.info("""
    **Partis Politiques:**
    - PPE-DE
    - PSE
    - ELDR
    - Verts/ALE
    - GUE-NGL
    """)

# Page d'accueil

if page == "Accueil":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## Bienvenue dans l'analyseur de discours politiques !")
        st.markdown("""
        Cette application interactive vous permet d'explorer les données parlementaires européennes entre 1999 et 2004,
        d'entraîner divers classifieurs pour prédire l'affiliation politique des députés,
        et de visualiser les résultats.
        
        ### Fonctionnalités principales
        
        1. **Analyse & Entraînement**
            - Chargement automatique des données.
            - Entraînement de plusieurs modèles de classification.
            - Évaluation des performances des modèles avec diverses métriques.
        
        &nbsp;
        
        2. **Prédiction**
            - Prédiction de l'affiliation politique.
            - Voir les probabilités associées.
            - Test avec plusieurs classifieurs.
        
        &nbsp;
        
        3. **Visualisations**
            - Graphiques interactifs des performances des modèles.
            - Matrices de confusion.
            - Comparaison des métriques.  
            
        ### Données
        - **Source**: Discours parlementaires européens (1999-2004).
        - **Documents**: ~19,000 (train) + ~5100 (test). [unilingue] / 58k (train) + 15k (test) [multilingue]
        - **Langues**: Français, Anglais, Italien.
        - **Classes**: 5 partis politiques principaux.
        
        """)
        
        if st.button("Commencer l'analyse", type="primary"):
            st.session_state.page = "Analyse & Entraînement"
            st.rerun()
    
    with col2:
        st.metric("Documents", "74k+")
        st.metric("Langues", "3")
        st.metric("Partis", "5")
        st.metric("Modèles", "6")

# Page d'analyse et d'entraînement
elif page == "Analyse & Entraînement":
    st.header("Analyse & Entraînement des Classifieurs")
    
    # Vérifier si la langue a changé
    language_changed = st.session_state.loaded_language != language
    
    if language_changed and st.session_state.data_loaded:
        st.warning(f"Vous avez changé de langue ({st.session_state.loaded_language} → {language}). Les données et modèles actuels ne correspondent plus. Veuillez recharger les données.")
        if st.button("Recharger avec la nouvelle langue", type="primary"):
            # Réinitialiser l'état
            st.session_state.data_loaded = False
            st.session_state.trained_models = {}
            st.session_state.results = {}
            st.session_state.predictions = {}
            st.session_state.loaded_language = None
            st.rerun()
    
    # Bouton de chargement des données
    if not st.session_state.data_loaded:
        if st.button("Charger les données", type="primary"):
            with st.spinner("Chargement des données..."):
                loader = DataLoader(".")
                
                if language == "all":
                    X_train, y_train, langs_train = loader.load_train_all_languages()
                    X_test, y_test, langs_test = loader.load_test_all_languages()
                else:
                    X_train, y_train, _ = loader.load_train_dataset(language)
                    X_test, y_test, _ = loader.load_test_dataset(language)
                    langs_train = [language] * len(X_train)
                    langs_test = [language] * len(X_test)
                
                # Filtrer les valeurs nulles
                valid_indices = [i for i, y in enumerate(y_test) if y is not None]
                X_test = [X_test[i] for i in valid_indices]
                y_test = [y_test[i] for i in valid_indices]
                langs_test = [langs_test[i] for i in valid_indices]
                
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.data_loaded = True
                st.session_state.loaded_language = language  # Enregistrer la langue chargée
                
                st.success(f"Données chargées ({language}): {len(X_train)} échantillons pour l'entraînement, {len(X_test)} pour le test.")
                st.rerun()
    
    if st.session_state.data_loaded:
        # Affichage des statistiques des données
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Documents entraînement", len(st.session_state.X_train))
        with col2:
            st.metric("Documents test", len(st.session_state.X_test))
        with col3:
            st.metric("Partis uniques", len(set(st.session_state.y_train)))
        with col4:
            st.metric("Langue", {"fr": "Français", "en": "Anglais",
                        "it": "Italien", "all": "Toutes"}[language])
        st.divider()
        
        # Distribution des classes
        st.subheader("Distribution des partis politiques")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Corpus d'entraînement**")
            train_dist = pd.Series(st.session_state.y_train).value_counts()
            st.bar_chart(train_dist)
            
        with col2:
            st.write("**Corpus de test**")
            test_dist = pd.Series(st.session_state.y_test).value_counts()
            st.bar_chart(test_dist)
        
        st.divider()
        
        # Entraînement des classifieurs
        st.subheader("Entraînement des classifieurs")
        
        if st.button("Lancer l'entraînement", type="primary"):
            # Récupérer les classifieurs
            classifiers = ClassifierFactory.get_classifiers()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            predictions = {}
            evaluator = EvaluationMetrics()
            
            for idx, (model_name, model_pipeline) in enumerate(classifiers.items()):
                status_text.text(f"Entraînement du modèle: {model_name}...")
                
                start_time = time.time()
                
                # Entraîner le modèle
                model_pipeline.fit(st.session_state.X_train, st.session_state.y_train)
                
                # Prédictions
                y_pred = model_pipeline.predict(st.session_state.X_test)
                
                # Évaluation
                metrics = evaluator.calculate_metrics(st.session_state.y_test, y_pred)
                
                elapsed = time.time() - start_time
                metrics['training_time'] = elapsed
                
                # Stocker les résultats
                results[model_name] = metrics
                predictions[model_name] = y_pred
                
                # Sauvegarder le modèle entraîné
                st.session_state.trained_models[model_name] = model_pipeline
                
                # Mettre à jour la barre de progression
                progress_bar.progress((idx + 1) / len(classifiers))
            
            st.session_state.results = results
            st.session_state.predictions = predictions
            
            status_text.empty()
            progress_bar.empty()
            st.success("Entraînement terminé !")
            st.rerun()
            
        # Afficher les résultats
        if st.session_state.results:
            st.divider()
            st.subheader("Résultats comparatifs des classifieurs")
            
            # Créer un DataFrame des résultats
            results_df = pd.DataFrame(st.session_state.results).T
            results_df = results_df.round(4)
            results_df = results_df.sort_values(by='f1_score', ascending=False)
            
            # Afficher le tableau des résultats
            st.dataframe(
                results_df[['accuracy', 'precision', 'recall', 'f1_score', 'training_time']],
                width='stretch'
            )
            
            # Meilleur modèle
            best_model = results_df.iloc[0]
            st.success(f"Meilleur modèle: **{best_model.name}** avec un F1-Score de **{best_model['f1_score']}**")
            
            # Graphique de comparaison
            st.subheader("Comparaison des performances des modèles")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(results_df))
            width = 0.2
            
            ax.bar(x - 1.5*width, results_df['accuracy'], width, label='Accuracy', alpha=0.8)
            ax.bar(x - 0.5*width, results_df['precision'], width, label='Precision', alpha=0.8)
            ax.bar(x + 0.5*width, results_df['recall'], width, label='Recall', alpha=0.8)
            ax.bar(x + 1.5*width, results_df['f1_score'], width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('Modèles', fontweight='bold')
            ax.set_ylabel('Scores', fontweight='bold')
            ax.set_title('Comparaison des performances', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(results_df.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            st.pyplot(fig)

# Page de prédiction
elif page == "Prédiction":
    st.header("Prédiction des Partis Politiques")
    
    if not st.session_state.trained_models:
        st.warning("Veuillez d'abord entraîner les modèles dans la section 'Analyse & Entraînement'.")
    else:
        st.info("Entrez un texte parlementaire pour prédire son affiliation politique.")
        
        # Sélection du modèle
        selected_model = st.selectbox(
            "Choisir un classifieur:",
            list(st.session_state.trained_models.keys())
        )
        
        # Zone de texte pour l'entrée utilisateur
        text_input = st.text_area(
            "Texte parlementaire:",
            height=200,
            placeholder="Entrez le texte ici..."
        )
        
        # Exemples
        if st.button("Charger un exemple"):
            if st.session_state.data_loaded:
                example_idx = np.random.randint(0, len(st.session_state.X_test))
                text_input = st.session_state.X_test[example_idx]
                st.session_state.example_text = text_input
                st.session_state.example_true = st.session_state.y_test[example_idx]
                st.rerun()
        
        if 'example_text' in st.session_state:
            text_input = st.session_state.example_text
            st.text_area(
                "Texte parlementaire:",
                value=text_input,
                height=200,
                disabled=True
            )
        
        # Prédiction
        if st.button("Prédire", type="primary") and text_input:
            with st.spinner("Prédiction en cours..."):
                model = st.session_state.trained_models[selected_model]
                
                # Prédiction et probabilités (si disponible)
                prediction = model.predict([text_input])[0]
                
                try:
                    proba = model.predict_proba([text_input])[0]
                    classes = model.classes_
                except:
                    proba = None
                
                st.divider()
                
                # Résultats
                col1, col2 = st.columns([1,2])
                
                with col1:
                    st.metric("Parti Prédit", prediction)
                    
                    if 'example_true' in st.session_state:
                        true_label = st.session_state.example_true
                        st.metric("Parti Réel", true_label)
                        
                        if prediction == true_label:
                            st.success("Prédiction correcte !")
                        else:
                            st.error("Prédiction incorrecte.")
                
                with col2:
                    if proba is not None:
                        st.write("**Probabilités par classe:**")
                        
                        # Créer un DataFrame pour les probabilités
                        proba_df = pd.DataFrame({
                            'Parti': classes,
                            'Probabilité': proba
                        }).sort_values(by='Probabilité', ascending=False)
                        
                        # Afficher avec une barre de progression
                        for _, row in proba_df.iterrows():
                            st.write(f"**{row['Parti']}**")
                            st.progress(float(row['Probabilité']))
                            st.write(f"{row['Probabilité']:.2%}")

# Page de visualisations
elif page == "Visualisations":
    st.header("Visualisations des Résultats")
    
    if not st.session_state.results:
        st.warning("Veuillez d'abord entraîner les modèles dans la section 'Analyse & Entraînement'.")
    else:
        vis_type = st.selectbox(
            "Type de visualisation:",
            ["Comparaison des métriques", "Matrice de Confusion", "Performance par Classe"]
        )
        
        if vis_type == "Comparaison des métriques":
            st.subheader("Comparaison des métriques des modèles")
            
            results_df = pd.DataFrame(st.session_state.results).T
            
            metrics_to_plot = st.multiselect(
                "Sélectionner les métriques à afficher:",
                ['accuracy', 'precision', 'recall', 'f1_score', 'f1_score_macro', 'precision_macro', 'recall_macro'],
                default=['accuracy', 'f1_score']
            )
            
            if metrics_to_plot:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                results_df[metrics_to_plot].plot(kind='bar', ax=ax, alpha=0.8)
                
                ax.set_title("Comparaison des métriques des modèles", fontweight='bold', fontsize=14)
                ax.set_xlabel("Modèles", fontweight='bold')
                ax.set_ylabel("Scores", fontweight='bold')
                ax.legend(title="Métriques")
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                st.pyplot(fig)
        
        elif vis_type == "Matrice de Confusion":
            st.subheader("Matrice de Confusion")
            
            selected_model = st.selectbox(
                "Choisir un classifieur:",
                list(st.session_state.predictions.keys())
            )
            
            cm = confusion_matrix(
                st.session_state.y_test,
                st.session_state.predictions[selected_model]
            )
            labels = sorted(list(set(st.session_state.y_test)))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f"Matrice de Confusion - {selected_model}", fontweight='bold', fontsize=14)
            ax.set_xlabel("Prédit", fontweight='bold')
            ax.set_ylabel("Réel", fontweight='bold')
            
            st.pyplot(fig)
        
        elif vis_type == "Performance par Classe":
            st.subheader("Performance par Classe")
            
            selected_model = st.selectbox(
                "Choisir un classifieur:",
                list(st.session_state.predictions.keys())
            )
            
            evaluator = EvaluationMetrics()
            per_class = evaluator.per_class_metrics(
                st.session_state.y_test,
                st.session_state.predictions[selected_model]
            )
            
            st.dataframe(per_class, width='stretch')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(per_class))
            width = 0.25
            
            ax.bar(x - width, per_class['Precision'], width, label='Precision', alpha=0.8)
            ax.bar(x, per_class['Recall'], width, label='Recall', alpha=0.8)
            ax.bar(x + width, per_class['F1-Score'], width, label='F1 Score', alpha=0.8)
            
            ax.set_title(f"Performance par Classe - {selected_model}", fontweight='bold', fontsize=14)
            ax.set_xlabel("Classes", fontweight='bold')
            ax.set_ylabel("Scores", fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(per_class['Classe'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)

# Page À propos
elif page == "À propos":
    st.header("À propos de cette application")
    
    st.markdown("""
    ### Contexte
    Cette application a été développée dans le cadre d'un projet d'analyse de discours parlementaires européens.
    Elle utilise des techniques de traitement du langage naturel (NLP) et d'apprentissage automatique pour prédire
    l'affiliation politique des députés basées sur leurs discours.
    
    ### Données
    Les données proviennent des discours parlementaires européens entre 1999 et 2004,
    couvrant trois langues principales: français, anglais et italien.
    
    ### Technologies Utilisées
    - **Langage**: Python
    - **Framework Web**: Streamlit
    - **Bibliothèques ML**: scikit-learn, pandas, numpy
    - **Visualisation**: Matplotlib, Seaborn
    
    ### Modèles utilisés
    - Naive Bayes (TF-IDF et CountVectorizer)
    - Complement Naive Bayes
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Decision Tree
    
    ### Auteurs
    - [Jean-Charles da Silva](https://github.com/jcharlesDS)
    
    ### Crédits
    Ce projet utilise des données publiques (DEFT'09, discours parlementaires européens) et des bibliothèques open-source.
    """)

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p> Prédiction des Partis Politiques - Décembre 2025/Janvier 2026 </p>
    </div>
""", unsafe_allow_html=True)


