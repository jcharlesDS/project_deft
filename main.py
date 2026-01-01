"""
Script principal pour l'analyse de prédiction des partis politiques.
Orchestre l'ensemble de la pipeline : Chargement des données, entraînement des modèles,
évaluation et visualisation.
"""

import os
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # Ignorer les avertissements pour une sortie plus propre

from data_loader import DataLoader
from classifiers import ClassifierFactory
from evaluation_metrics import EvaluationMetrics
from visualizer import Visualizer

def main():
    print("~"*80)
    print("Prédiction des Partis Politiques")
    print("~"*80)
    print()
    
    # Dossier pour les résultats
    os.makedirs("resultats", exist_ok=True)
    os.makedirs("resultats/graphiques", exist_ok=True)
    
    # Chargement des données
    print("1. Chargement des données...")
    print("~"*40)
    
    loader = DataLoader(".")
    
    # Charger les données d'entraînement
    print("Chargement des données d'entraînement (FR, EN, IT)...")
    X_train, y_train, _ = loader.load_train_all_languages()
    print(f" {len(X_train)} échantillons chargés pour l'entraînement.")
    
    # Charger les données de test
    print("Chargement des données de test (FR, EN, IT)...")
    X_test, y_test, langs_test = loader.load_test_all_languages()
    
    # Filtrer les valeurs nulles (documents sans référence, si il y en a)
    valid_indices = [i for i, y in enumerate(y_test) if y is not None]
    X_test = [X_test[i] for i in valid_indices]
    y_test = [y_test[i] for i in valid_indices]
    langs_test = [langs_test[i] for i in valid_indices]
    
    print(f" {len(X_test)} échantillons chargés pour le test.")
    
    # Statistiques des données
    print("\nStatistiques des données :")
    print(f"\n Partis politiques uniques: {sorted(set(y_train))}")
    print(f" Distribution dans le corpus d'entraînement/apprentissage :")
    train_distrib = pd.Series(y_train).value_counts()
    for parti, count in train_distrib.items():
        print(f"  - {parti}: {count} échantillons")
    
    print()
    
    # Visualisation de la distribution des classes
    print("2. Visualisation de la distribution des classes...")
    print("~"*40)
    
    vis = Visualizer()
    vis.plot_class_distribution(y_train, y_test, save_path="resultats/graphiques/distribution_classes.png")
    
    # Entraînement et évaluation des classificateurs
    print("\n3. Entraînement et évaluation des classificateurs...")
    print("~"*40)
    
    classifiers = ClassifierFactory.get_classifiers()
    print(f"Classificateurs à évaluer: {list(classifiers.keys())}\n")
    
    # Stocker les résultats
    all_results = {}
    all_predictions = {}
    
    evaluator = EvaluationMetrics()
    
    for model_name, model_pipeline in classifiers.items():
        print(f"Entraînement du modèle: {model_name}...")
        start_time = time.time()
        
        try:
            # Entraînements
            print(" Entraînement en cours...")
            model_pipeline.fit(X_train, y_train)
            
            # Prédictions
            print(" Prédiction en cours...")
            y_pred = model_pipeline.predict(X_test)
            all_predictions[model_name] = y_pred
            
            # Évaluation
            print(" Évaluation en cours...")
            metrics = evaluator.calculate_metrics(y_test, y_pred)
            
            elapsed = time.time() - start_time
            metrics['training_time'] = elapsed  # Ajouter le temps d'entraînement aux métriques
            
            all_results[model_name] = metrics
            
            print(f" Modèle {model_name} évalué en {elapsed:.2f} secondes.\n")
            print(f" Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}\n")
        except Exception as e:
            print(f" Échec de l'évaluation du modèle {model_name}: {e}\n")
            continue
    
    print()
    
    # Comparaison des modèles
    print("4. Comparaison des modèles...")
    print("~"*40)
    
    # Créer un DataFrame comparatif
    results_df = evaluator.compare_models(all_results)
    
    # Trier par f1_score si la colonne existe
    if 'f1_score' in results_df.columns:
        results_df = results_df.sort_values('f1_score', ascending=False)
    
    # Sauvegarder les résultats
    results_df.to_csv("resultats/comparaison_modeles.csv", index=False, sep=';') # Changer le séparateur en ',' si anglophone
    print("Résultats sauvegardés dans 'resultats/comparaison_modeles.csv'\n")
    
    # Afficher le tableau comparatif
    print('\n Tableau comparatif des modèles:')
    print(" " + "~"*76)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.precision', 4)
    
    # Afficher les colonnes disponibles ou un message si le DataFrame est vide
    if not results_df.empty and 'Model' in results_df.columns:
        display_cols = [col for col in ['Model', 'accuracy', 'precision', 'recall', 'f1_score'] if col in results_df.columns]
        print(results_df[display_cols].to_string(index=False))
        print()
        
        if not results_df.empty:
            best_model = results_df.iloc[0] # Afficher le meilleur modèle
            print(f"Meilleur modèle: {best_model['Model']}")
    else:
        print("Aucun modèle n'a pu être évalué avec succès.")
    print(f" Accuracy: {best_model['accuracy']:.4f}")
    print(f" Precision: {best_model['precision']:.4f}")
    print(f" Recall: {best_model['recall']:.4f}")
    print(f" F1-Score: {best_model['f1_score']:.4f}")
    
    print()
    
    # Générer des graphiques de performance
    print("5. Génération des graphiques de performance...")
    print("~"*40)
    
    # Comparaison des métriques principales
    print(" Comparaison des métriques principales...")
    vis.plot_metrics_comparison(results_df, save_path="resultats/graphiques/comparaison_modeles.png")
    
    # Graphique détaillé
    print(" Comparaison détaillée des modèles...")
    vis.plot_detailed_comparison(results_df, save_path="resultats/graphiques/comparaison_detaillee_modeles.png")
    
    # Top 5 modèles
    print(" Top 5 modèles...")
    vis.plot_top_models(results_df, top_n=5, save_path="resultats/graphiques/top_5_modeles.png")
    
    # Matrice de confusion du meilleur modèle
    print(f" Matrice de confusion pour le meilleur modèle: {best_model['Model']}...")
    best_model_name = best_model['Model']
    vis.plot_confusion_matrix(y_test, all_predictions[best_model_name],
            best_model_name, save_path="resultats/graphiques/matrice_confusion_meilleur_modele_multilingue.png")
    
    print(" Tous les graphiques ont été sauvegardés dans le dossier 'resultats/graphiques/'.")
    print()
    
    # Rapport détaillé pour le meilleur modèle
    print("6. Rapport détaillé pour le meilleur modèle...")
    print("~"*40)
    
    evaluator.print_summary(best_model_name, all_results[best_model_name], y_test, all_predictions[best_model_name])
    
    # Analyse par langue
    print("\n7. Analyse par langue...")
    print("~"*40)
    
    results_by_lang = {}
    
    for lang, lang_name in [('fr', 'Français'), ('en', 'Anglais'), ('it', 'Italien')]:
        print(f" Analyse pour la langue: {lang_name}...")
        
        # Charger les données spécifiques à la langue
        x_train_lang, y_train_lang, _ = loader.load_train_dataset(lang)
        X_test_lang, y_test_lang, _ = loader.load_test_dataset(lang)
        
        # Filtrer les valeurs nulles
        valid_indices = [i for i, y in enumerate(y_test_lang) if y is not None]
        X_test_lang = [X_test_lang[i] for i in valid_indices]
        y_test_lang = [y_test_lang[i] for i in valid_indices]
        
        if not X_test_lang:
            print(f"  Aucun échantillon de test pour la langue {lang_name}. Passage à la suivante.\n")
            continue
        
        print(f" Entraînement {len(x_train_lang)}, Test: {len(X_test_lang)} échantillons.")
        
        # Tester tous les modèles sur chaque langue
        lang_results = {}
        
        for model_name in results_df.head(3)['Model']:
            model_pipeline = ClassifierFactory.get_classifiers()[model_name]
            
            # Entraînement et prédiction
            model_pipeline.fit(x_train_lang, y_train_lang)
            y_pred_lang = model_pipeline.predict(X_test_lang)
            
            # Évaluation
            metrics = evaluator.calculate_metrics(y_test_lang, y_pred_lang)
            lang_results[model_name] = metrics

        # Créer un DataFrame pour les résultats par langue
        lang_df = evaluator.compare_models(lang_results)
        results_by_lang[lang] = lang_df
        
        # Afficher le meilleur modèle pour la langue
        best_lang = lang_df.iloc[0]
        print(f"  Meilleur modèle pour {lang_name}: {best_lang['Model']} avec F1-Score: {best_lang['f1_score']:.4f}\n")

    # Graphique de comparaison par langue
    print(" Graphique de comparaison des meilleurs modèles par langue...")
    vis.plot_per_language_comparison(results_by_lang, save_path="resultats/graphiques/comparaison_par_langue.png")
    
    print()
    
    # Sauvegarder le rapport détaillé complet
    print("8. Génération du rapport détaillé complet...")
    print("~"*40)
    
    with open('resultats/rapport_detaille.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT DÉTAILLÉ - PRÉDICTION DES PARTIS POLITIQUES\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: Meilleur modèle sur corpus multilingue
        f.write("~"*80 + "\n")
        f.write("SECTION 1: MEILLEUR MODÈLE SUR CORPUS MULTILINGUE (FR + EN + IT)\n")
        f.write("~"*80 + "\n\n")
        f.write(f"Modèle: {best_model_name}\n")
        f.write(f"Corpus d'entraînement: {len(X_train)} échantillons (toutes langues)\n")
        f.write(f"Corpus de test: {len(y_test)} échantillons (toutes langues)\n\n")
        
        f.write("Métriques globales:\n")
        f.write(f"  • Accuracy: {all_results[best_model_name]['accuracy']:.4f}\n")
        f.write(f"  • Precision (Weighted): {all_results[best_model_name]['precision']:.4f}\n")
        f.write(f"  • Recall (Weighted): {all_results[best_model_name]['recall']:.4f}\n")
        f.write(f"  • F1-Score (Weighted): {all_results[best_model_name]['f1_score']:.4f}\n\n")
        
        f.write("Métriques Macro:\n")
        f.write(f"  • Precision (Macro): {all_results[best_model_name]['precision_macro']:.4f}\n")
        f.write(f"  • Recall (Macro): {all_results[best_model_name]['recall_macro']:.4f}\n")
        f.write(f"  • F1-Score (Macro): {all_results[best_model_name]['f1_score_macro']:.4f}\n\n")
        
        f.write("Rapport de classification détaillé:\n")
        f.write("-" * 80 + "\n")
        f.write(evaluator.generate_classification_report(y_test, all_predictions[best_model_name]))
        f.write("\n")
        
        # Section 2: Résultats par langue
        f.write("\n" + "~"*80 + "\n")
        f.write("SECTION 2: ANALYSE PAR LANGUE\n")
        f.write("~"*80 + "\n\n")
        
        lang_names = {'fr': 'Français', 'en': 'Anglais', 'it': 'Italien'}
        
        for lang in ['fr', 'en', 'it']:
            if lang in results_by_lang:
                f.write("=" * 80 + "\n")
                f.write(f"LANGUE: {lang_names[lang].upper()} ({lang})\n")
                f.write("=" * 80 + "\n\n")
                
                lang_df = results_by_lang[lang]
                best_lang_model = lang_df.iloc[0]
                
                f.write(f"Meilleur modèle pour cette langue: {best_lang_model['Model']}\n\n")
                
                f.write("Comparaison des top 3 modèles:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Modèle':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
                f.write("-" * 80 + "\n")
                
                for idx, row in lang_df.head(3).iterrows():
                    f.write(f"{row['Model']:<35} {row['accuracy']:>10.4f} {row['precision']:>10.4f} "
                        f"{row['recall']:>10.4f} {row['f1_score']:>10.4f}\n")
                
                f.write("\n")
        
        # Section 3: Comparaison globale
        f.write("\n" + "~"*80 + "\n")
        f.write("SECTION 3: COMPARAISON TOUS MODÈLES (CORPUS MULTILINGUE)\n")
        f.write("~"*80 + "\n\n")
        
        f.write(f"{'Modèle':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Temps (s)':>10}\n")
        f.write("=" * 100 + "\n")
        
        for idx, row in results_df.iterrows():
            f.write(f"{row['Model']:<35} {row['accuracy']:>10.4f} {row['precision']:>10.4f} "
                f"{row['recall']:>10.4f} {row['f1_score']:>10.4f} {row.get('training_time', 0):>10.2f}\n")
        
        f.write("\n")
        
        # Section 4: Conclusion
        f.write("\n" + "~"*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("~"*80 + "\n\n")
        f.write(f"• Meilleur modèle global (multilingue): {best_model['Model']} "
            f"(F1-Score: {best_model['f1_score']:.4f})\n")
        
        for lang in ['fr', 'en', 'it']:
            if lang in results_by_lang:
                best_lang = results_by_lang[lang].iloc[0]
                f.write(f"• Meilleur modèle pour {lang_names[lang]}: {best_lang['Model']} "
                    f"(F1-Score: {best_lang['f1_score']:.4f})\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Fin du rapport\n")
        f.write("=" * 80 + "\n")
    
    print("Rapport détaillé complet sauvegardé dans 'resultats/rapport_detaille.txt'.")
    print()
    
    # Résumé final
    print("~"*80)
    print("Analyse terminée.")
    print("~"*80)
    print(f"\nRésultats disponibles dans le dossier 'resultats/':")
    print(" - comparaison_modeles.csv")
    print(" - rapport_detaille.txt")
    print(" - graphiques/ (dossier avec tous les graphiques générés)")
    print(f"\n Meilleur modèle global: {best_model['Model']} avec F1-Score: {best_model['f1_score']:.4f}")
    print("~"*80)

if __name__ == "__main__":
    main()