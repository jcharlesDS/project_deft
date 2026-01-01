"""
Module de visualisation.
Génère des graphiques pour évaluer les performances des modèles de classification.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import confusion_matrix

class Visualizer:
    """ Classe pour la visualisation des performances des modèles de classification."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Création de l'instance de la classe Visualizer.
        
        Args:
            style (str, optional): Style à utiliser pour les graphiques. Ici, 'seaborn-v0_8-darkgrid'.
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_metrics_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Génère des graphiques comparant les métriques principales de performance des modèles.
        
        Args:
            results_df (pd.DataFrame): DataFrame contenant les résultats des modèles avec les colonnes 'Model', 'accuracy', 'precision', 'recall', 'f1_score'.
            save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, le graphique est affiché.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparaison des métriques de performance des modèles', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Trier par métrique
            data = results_df.sort_values(metric, ascending=True)
            
            # Créer le graphique à barres horizontales
            bars = ax.barh(data['Model'], data[metric])
            
            # Définir les couleurs en fonction des valeurs
            colors = plt.cm.RdYlGn(data[metric])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Ajouter les annotations des valeurs
            for i, v in enumerate(data[metric]):
                ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontweight='bold')
            
            ax.set_xlabel(name, fontweight='bold')
            ax.set_title(f'{name} par modèle', fontweight='bold')
            ax.set_xlim(0, 1.1)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detailed_comparison(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Génère des graphiques détaillés comparant TOUTES les métriques de performance des modèles. (weighted, macro, micro, etc.)
        
        Args:
            results_df (pd.DataFrame): DataFrame contenant les résultats des modèles avec les colonnes 'Model', 'accuracy', 'precision', 'recall', 'f1_score'.
            save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, le graphique est affiché.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sélectionner les métriques à tracer
        metrics_to_plot = ['accuracy', 'f1_score', 'f1_score_macro', 'f1_score_micro', 'precision', 'recall']
        
        x = np.arange(len(results_df))  # la position des groupes
        width = 0.12  # la largeur des barres
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in results_df.columns:
                offset = width * (i - len(metrics_to_plot) / 2)
                ax.bar(x + offset, results_df[metric], width, label=metric.replace('_', ' ').title(), alpha=0.8)
    
        ax.set_xlabel('Modèles', fontweight='bold', fontsize=12)
        ax.set_ylabel('Scores', fontweight='bold', fontsize=12)
        ax.set_title('Comparaison détaillée des métriques', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], class_names: List[str], save_path: str = None):
        """
        Génère une matrice de confusion pour évaluer les performances d'un modèle de classification.
        
        Args:
            y_true (List[str]): Liste des étiquettes vraies.
            y_pred (List[str]): Liste des étiquettes prédites par le modèle.
            class_names (List[str]): Liste des noms des modèles.
            save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, le graphique est affiché.
        """
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Nombre de prédictions'}, ax=ax)
        
        ax.set_title(f'Matrice de Confusion - {class_names}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Étiquettes Prédites', fontweight='bold')
        ax.set_ylabel('Étiquettes Réelles', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_models(self, results_df: pd.DataFrame, top_n: int = 5, metric: str = 'f1_score', save_path: str = None):
        """
        Affiche les meilleurs modèles selon une métrique spécifique.
        
        Args:
            results_df (pd.DataFrame): DataFrame contenant les résultats des modèles avec les colonnes 'Model' et la métrique spécifiée.
            top_n (int, optional): Nombre de modèles à afficher. Par défaut, 5.
            metric (str, optional): La métrique à utiliser pour le classement. Par défaut, 'f1_score'.
            save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, le graphique est affiché.
        """
        top_data = results_df.nlargest(top_n, metric)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(top_data))
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score']
        width = 0.2
        
        for i, met in enumerate(metrics_to_show):
            if met in top_data.columns:
                offset = width * (i - len(metrics_to_show) / 2 + 0.5)
                ax.bar(x + offset, top_data[met], width, label=met.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Modèles', fontweight='bold', fontsize=12)
        ax.set_ylabel('Scores', fontweight='bold', fontsize=12)
        ax.set_title(f'Top {top_n} Modèles (par {metric})', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(top_data['Model'], rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_language_comparison(self, results_by_lang: Dict[str, pd.DataFrame], save_path: str = None):
        """
        Génère des graphiques comparant les performances des modèles par langue.
        
        Args:
            results_by_lang (Dict[str, pd.DataFrame]): Dictionnaire où les clés sont les langues et les valeurs sont des DataFrames contenant les résultats des modèles pour chaque langue.
            save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, le graphique est affiché.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Comparaison des performances des modèles par langue', fontsize=16, fontweight='bold')
        
        languages = ['fr', 'en', 'it']
        lang_names = {'fr': 'Français', 'en': 'Anglais', 'it': 'Italien'}
        
        for idx, lang in enumerate(languages):
            ax = axes[idx]
            
            if lang in results_by_lang:
                data = results_by_lang[lang]
                
                # Top 5 modèles pour chaque langue
                top_5 = data.nlargest(5, 'f1_score')
                
                x = np.arange(len(top_5))
                width = 0.35
                
                ax.bar(x - width/2, top_5['f1_score'], width, label='F1 Score', alpha=0.8)
                ax.bar(x + width/2, top_5['accuracy'], width, label='Accuracy', alpha=0.8)
                
                ax.set_title(lang_names[lang], fontweight='bold', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(top_5['Model'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_train: List[str], y_test: List[str], save_path: str = None):
        """
        Affiche la distribution des classes dans les ensembles d'entraînement et de test.
        
        Args:
            y_train (List[str]): Liste des étiquettes de l'ensemble d'entraînement.
            y_test (List[str]): Liste des étiquettes de l'ensemble de test.
            save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, le graphique est affiché.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Distribution des partis politiques', fontsize=16, fontweight='bold')
        
        # Distribution des classes dans l'ensemble d'entraînement
        train_counts = pd.Series(y_train).value_counts().sort_index()
        ax1.bar(range(len(train_counts)), train_counts.values, alpha=0.7)
        ax1.set_xticks(range(len(train_counts)))
        ax1.set_xticklabels(train_counts.index, rotation=45, ha='right')
        ax1.set_title('Ensemble d\'entraînement', fontweight='bold')
        ax1.set_ylabel('Nombre d\'échantillons', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Annotations des valeurs
        for i, v in enumerate(train_counts.values):
            ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # Distribution des classes dans l'ensemble de test
        test_counts = pd.Series(y_test).value_counts().sort_index()
        ax2.bar(range(len(test_counts)), test_counts.values, alpha=0.7, color='orange')
        ax2.set_xticks(range(len(test_counts)))
        ax2.set_xticklabels(test_counts.index, rotation=45, ha='right')
        ax2.set_title('Ensemble de test', fontweight='bold')
        ax2.set_ylabel('Nombre d\'échantillons', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Annotations des valeurs
        for i, v in enumerate(test_counts.values):
            ax2.text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()