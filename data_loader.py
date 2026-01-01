"""
Module chargeant les données XML et les préparant pour la tache de classification.
"""

import xml.etree.ElementTree as ET
import re
from typing import List, Tuple, Dict

class DataLoader:
    """Chargement des données pour la classification"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        
    def parse_xml(self, xml_path: str) -> List[Dict]:
        """Parse un fichier XML et extrait les documents avec leurs informations.
        
        Args:
            xml_path (str): Chemin vers le fichier XML.
            
        Returns:
            List[Dict]: Liste de dictionnaires contenant les informations des documents.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        documents = []
        
        for doc in root.findall('doc'):
            doc_id = doc.get('id')
            
            # Extraction des informations
            # Exemple: 2_fr:1 -> lang=fr, file_id=1, corpus_id=2 
            # Pour les fichiers de test: juste un nombre simple (ex: "1", "2", etc.)
            
            # Tenter le format complet (apprentissage)
            match = re.match(r'(\d+)_([a-z]+):(\d+)', doc_id)
            if match:
                corpus_id, lang, file_id = match.groups()
            else:
                # Tenter le format du simple nombre (test)
                if doc_id and doc_id.isdigit():
                    corpus_id, lang, file_id = None, None, doc_id
                else:
                    corpus_id, lang, file_id = None, None, None
            
            # Extraction du parti politique
            parti = None
            eval_parti = doc.find('.//EVAL_PARTI/PARTI')
            if eval_parti is not None:
                parti = eval_parti.get('valeur')
            
            # Extraction du texte
            text_elem = doc.find('texte')
            texte = ""
            if text_elem is not None:
                texte = " ".join([p.text.strip() for p in text_elem.findall('p') if p.text])
            
            documents.append({
                'doc_id': doc_id,
                'corpus_id': corpus_id,
                'lang': lang,
                'file_id': file_id,
                'parti': parti,
                'texte': texte
            })
        return documents
    
    def load_reference(self, ref_path: str) -> Dict[str, str]:
        """
        Charge le fichier de référence des partis politiques.
        
        Args:
            ref_path (str): Chemin vers le fichier de référence. (.txt)
            
        Returns:
            Dict[str, str]: Dictionnaire mappant les codes de partis à leurs noms complets.
        """
        reference = {}
        with open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    file_id, parti = parts
                    reference[int(file_id)] = parti
        return reference
    
    def load_train_dataset(self, lang: str = 'fr') -> Tuple[List[str], List[str], List[Dict]]:
        """
        Charge le jeu de données d'entraînement pour une langue donnée.
        
        Args:
            lang (str): Langue des documents à charger ('fr', 'en', 'it').
            
        Returns:
            Tuple[List[str], List[str], List[Dict]]: Textes, labels (partis politiques), et les informations extraites des documents XML.
        """
        xml_path = f"{self.base_path}/apprentissage/deft09_parlement_appr_{lang}.xml"
        documents = self.parse_xml(xml_path)
        
        textes = [doc['texte'] for doc in documents]
        labels = [doc['parti'] for doc in documents]
        
        return textes, labels, documents
    
    def load_test_dataset(self, lang: str = 'fr') -> Tuple[List[str], List[str], List[Dict]]:
        """
        Charge le jeu de données de test pour une langue donnée avec les références.
        
        Args:
            lang (str): Langue des documents à charger ('fr', 'en', 'it').
            
        Returns:
            Tuple[List[str], List[str], List[Dict]]: Textes, labels de référence (partis politiques), et les informations extraites des documents XML.
        """
        # Utiliser les fichiers nettoyés (sans doublons)
        xml_path = f"{self.base_path}/test/deft09_parlement_test_{lang}_cleaned.xml"
        ref_path = f"{self.base_path}/reference/deft09_parlement_ref_{lang}.txt"
        
        documents = self.parse_xml(xml_path) # Charger les documents de test (sans labels)
        reference = self.load_reference(ref_path) # Charger les labels de référence
        
        textes = []
        labels = []
        
        # Associer les labels de référence aux documents
        for doc in documents:
            textes.append(doc['texte'])
            
            # Convertir file_id en entier si possible
            file_id = None
            if doc['file_id'] is not None:
                try:
                    file_id = int(doc['file_id'])
                except (ValueError, TypeError):
                    file_id = None
            
            labels.append(reference.get(file_id, None))
        
        return textes, labels, documents
    
    def load_train_all_languages(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Charge les données d'entraînement pour toutes les langues (FR, EN, IT).
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Textes, labels et langues.
        """
        all_textes = []
        all_labels = [] 
        all_langs = []
        
        for lang in ['fr', 'en', 'it']:
            textes, labels, _ = self.load_train_dataset(lang)
            all_textes.extend(textes)
            all_labels.extend(labels)
            all_langs.extend([lang] * len(textes))
        
        return all_textes, all_labels, all_langs
    
    def load_test_all_languages(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Charge les données de test pour toutes les langues (FR, EN, IT).
        
        Returns:
            Tuple[List[str], List[str], List[str]]: Textes, labels et langues.
        """
        all_textes = []
        all_labels = [] 
        all_langs = []
        
        for lang in ['fr', 'en', 'it']:
            textes, labels, _ = self.load_test_dataset(lang)
            all_textes.extend(textes)
            all_labels.extend(labels)
            all_langs.extend([lang] * len(textes))
        
        return all_textes, all_labels, all_langs
