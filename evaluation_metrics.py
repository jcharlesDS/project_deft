"""
Module pour l'évaluation des performances des modèles de classification.
Il fournit des fonctions pour calculer diverses métriques telles que la précision,
le rappel, la F-mesure, l'accuracy. 
On y trouve également des fonctions pour générer
des rapports de classification et des matrices de confusion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class EvaluationMetrics:
    """
    Evaluation des performances des classifieurs
    """
    
    @staticmethod
    def calculate_metrics(y_true: List[str], y_pred: List[str], average: str = 'weighted') -> dict[str, float]:
        """
        Calcule les métriques de performance pour les modèles de classification.

        Args:
            y_true (List[str]): Les étiquettes vraies.
            y_pred (List[str]): Les étiquettes prédites par le modèle.
            average (str): Type de moyenne ('micro', 'macro', 'weighted')

        Returns:
            dict: Un dictionnaire contenant toutes les métriques calculées.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Métriques macro et micro
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_score_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_score_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        return metrics
    
    @staticmethod
    def generate_classification_report(y_true: List[str], y_pred: List[str]) -> str:
        """
        Génère un rapport de classification détaillé.

        Args:
            y_true (List[str]): Les étiquettes vraies.
            y_pred (List[str]): Les étiquettes prédites par le modèle.

        Returns:
            str: Rapport de classification sous forme de chaîne de caractères.
        """
        return classification_report(y_true, y_pred, zero_division=0)
    
    @staticmethod
    def generate_confusion_matrix(y_true: List[str], y_pred: List[str]) -> np.ndarray:
        """
        Génère la matrice de confusion.

        Args:
            y_true (List[str]): Les étiquettes vraies.
            y_pred (List[str]): Les étiquettes prédites par le modèle.

        Returns:
            np.ndarray: Matrice de confusion.
        """
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def per_class_metrics(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
        """
        Calcule les métriques de performance pour chaque classe.

        Args:
            y_true (List[str]): Les étiquettes vraies.
            y_pred (List[str]): Les étiquettes prédites par le modèle.

        Returns:
            pd.DataFrame: DataFrame contenant les métriques pour chaque classe.
        """
        
        # Obtenir la liste des classes uniques
        classes = sorted(list(set(y_true)))
        
        metrics_per_class = []
        
        
        for classe in classes:
            # Binariser les étiquettes pour la classe actuelle
            y_true_binary = [1 if label == classe else 0 for label in y_true]
            y_pred_binary = [1 if label == classe else 0 for label in y_pred]
            
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            support = sum(y_true_binary) # Nombre d'occurrences de la classe dans y_true
            
            metrics_per_class.append({
                'Classe': classe,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': support
            })
        
        return pd.DataFrame(metrics_per_class)
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare les performances de plusieurs modèles.

        Args:
            results (Dict[str, Dict[str, Any]]): Dictionnaire où les clés sont les noms des modèles
                                                    et les valeurs sont des dictionnaires de métriques.

        Returns:
            pd.DataFrame: DataFrame comparant les performances des modèles.
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            data = {'Model': model_name}
            data.update(metrics)
            comparison_data.append(data)
            
        df = pd.DataFrame(comparison_data)
        
        # Réorganiser les colonnes pour mettre les métriques principales en premier
        # Vérifier que les colonnes existent avant de réorganiser
        priority_cols = ['Model', 'accuracy', 'precision', 'recall', 'f1_score']
        existing_priority_cols = [col for col in priority_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_priority_cols]
        df = df[existing_priority_cols + other_cols]
        
        return df
    
    @staticmethod
    def print_summary(model_name: str, metrics: dict[str, float], y_true: List[str], y_pred: List[str]) -> None:
        """
        Affiche un résumé des performances des modèles.

        Args:
            model_name (str): Nom du modèle.
            metrics (dict): Dictionnaire des métriques calculées.
            y_true (List[str]): Les étiquettes vraies.
            y_pred (List[str]): Les étiquettes prédites par le modèle.
        """
        
        print(f"\n{'~' * 50}")
        print(f"Résumé des performances pour le modèle: {model_name}")
        print(f"{'~' * 50}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (weighted): {metrics['precision']:.4f}")
        print(f"Recall (weighted): {metrics['recall']:.4f}")
        print(f"F1-Score (weighted): {metrics['f1_score']:.4f}")
        print(f"\nPrecision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro): {metrics['f1_score_macro']:.4f}")
        print(f"\n{'~' * 50}")
        print("Rapport de classification:")
        print(f"{'~' * 50}")
        print(EvaluationMetrics.generate_classification_report(y_true, y_pred))
