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
        # Naive Bayes avec TF-IDF
        classifiers['Naive Bayes (TF-IDF)'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        
        # Complement Naive Bayes avec TF-IDF
        classifiers['Complement Naive Bayes'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95
            )),
            ('clf', ComplementNB(alpha=0.1))
        ])
        
        # Naive Bayes avec CountVectorizer
        classifiers['Naive Bayes (Count)'] = Pipeline([
            ('count', CountVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        
        # Logistic Regression avec TF-IDF
        classifiers['Logistic Regression (TF-IDF)'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95
            )),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        # Linear SVM
        classifiers['Linear SVM'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95
            )),
            ('clf', LinearSVC(
                C=1.0, 
                max_iter=2000,
                random_state=42
            ))
        ])
        
        # Decision Tree
        classifiers['Decision Tree'] = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000, 
                ngram_range=(1, 2),
                min_df=2, 
                max_df=0.95
            )),
            ('clf', DecisionTreeClassifier(
                max_depth=15, 
                min_samples_split=10,
                random_state=42
            ))
        ])
        
        return classifiers
