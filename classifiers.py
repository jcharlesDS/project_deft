"""
Module contenant les différents classifieurs pour la prédiction des labels.
"""

from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def get_stopwords():
    """
    Récupère les stopwords pour les langues du projet (FR, EN, IT)
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        import ssl
        
        # Contourner les problèmes SSL si nécessaire
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Télécharger les ressources si nécessaire
        try:
            stopwords.words('french')
        except LookupError:
            print("Téléchargement des stopwords NLTK...")
            nltk.download('stopwords', quiet=True)
            print("Stopwords téléchargés")
        
        # Combiner les stopwords de toutes les langues du projet
        stop_words = set()
        stop_words.update(stopwords.words('french'))
        stop_words.update(stopwords.words('english'))
        stop_words.update(stopwords.words('italian'))
        
        print(f"✓ {len(stop_words)} stopwords chargés (FR+EN+IT)")
        return list(stop_words)
    
    except ImportError:
        print("NLTK non installé. Les stopwords ne seront pas utilisés.")
        print("   Installez NLTK avec: pip install nltk")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des stopwords: {e}")
        return None


class ClassifierFactory:
    """
    Usine à classifieurs pour la prédiction des labels.
    """

    @staticmethod
    def get_classifiers() -> Dict[str, Pipeline]:
        """
        Retourne un dictionnaire de classifieurs avec leurs pipelines associés.
        
        Returns:
            Dict[str, Pipeline]: Dictionnaire des classifieurs.
        """
        classifiers = {}
        stop_words = get_stopwords()
        
        # Naive Bayes avec TF-IDF
        classifiers['Naive Bayes (TF-IDF)'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=12000,  # Augmenté pour améliorer la couverture des données
                ngram_range=(1, 3),  # Tri-grammes pour expressions politiques
                min_df=5,  # Filtrer mots très rares
                max_df=0.90,  # Plus strict sur mots fréquents
                sublinear_tf=True,  # Scaling logarithmique
                stop_words=stop_words
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        
        # Complement Naive Bayes avec TF-IDF
        classifiers['Complement Naive Bayes'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=12000,
                ngram_range=(1, 3),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                stop_words=stop_words
            )),
            ('clf', ComplementNB(alpha=0.1))
        ])
        
        # Naive Bayes avec CountVectorizer
        classifiers['Naive Bayes (Count)'] = Pipeline([
            ('count', CountVectorizer(
                max_features=12000,
                ngram_range=(1, 3),
                min_df=5,
                max_df=0.90,
                stop_words=stop_words
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        
        # Logistic Regression avec TF-IDF
        classifiers['Logistic Regression (TF-IDF)'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=15000,  # Augmentation pour meilleure performance
                ngram_range=(1, 3),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                stop_words=stop_words
            )),
            ('clf', LogisticRegression(
                max_iter=1500,
                C=2.0,  # Régularisation réduite
                class_weight='balanced'  # Gérer le déséquilibre
            ))
        ])
        
        # Linear SVM
        classifiers['Linear SVM'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 3),
                min_df=5,
                max_df=0.90,
                sublinear_tf=True,
                stop_words=stop_words
            )),
            ('clf', LinearSVC(
                C=1.5,  # Augmenté pour meilleure performance
                max_iter=3000,  # Plus d'itérations pour convergence
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Decision Tree
        classifiers['Decision Tree'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95,
                stop_words=stop_words
            )),
            ('clf', DecisionTreeClassifier(
                max_depth=15, 
                min_samples_split=10,
                random_state=42
            ))
        ])
        
        return classifiers

