"""
Script pour nettoyer les fichiers XML de test en supprimant les doublons avec le train
"""
import xml.etree.ElementTree as ET


def extract_textes_from_xml(xml_file):
    """Extrait tous les contenus des balises <texte> d'un fichier XML"""
    textes_dict = {}  # Dictionnaire pour stocker les textes et leurs doc_id
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for doc in root.findall('.//doc'):
            doc_id = doc.get('id', 'unknown')
            texte_elem = doc.find('texte')
            
            if texte_elem is not None:
                paragraphes = []
                for para in texte_elem.findall('p'):
                    if para.text:
                        texte = para.text.strip()
                        if texte:
                            paragraphes.append(texte)
                
                if paragraphes:
                    texte_complet = ' '.join(paragraphes)
                    textes_dict[texte_complet] = doc_id
        
        print(f"✓ {len(textes_dict)} documents uniques dans {xml_file}")
        return textes_dict
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return {}


def remove_duplicates(train_file, test_file, output_file):
    """Supprime les doublons du fichier test"""
    
    print("="*80)
    print("SUPPRESSION DES DOUBLONS DU FICHIER TEST")
    print("="*80)
    print()
    
    # Extraire les textes du train
    print("Extraction des textes du train...")
    train_textes = extract_textes_from_xml(train_file)
    
    # Extraire les textes du test
    print("Extraction des textes du test...")
    test_textes_dict = extract_textes_from_xml(test_file)
    
    print()
    
    # Identifier les doublons
    doublons = set()
    for texte in test_textes_dict.keys():
        if texte in train_textes:
            doublons.add(test_textes_dict[texte])
    
    print(f"Doublons trouvés: {len(doublons)}")
    print()
    
    if not doublons:
        print("✓ Aucun doublon détecté. Pas de modification nécessaire.")
        return
    
    # Parser le fichier test et supprimer les documents en doublon
    tree = ET.parse(test_file)
    root = tree.getroot()
    
    # Supprimer les documents doublons
    for doc in root.findall('doc'):
        doc_id = doc.get('id')
        if doc_id in doublons:
            root.remove(doc)
            print(f"  Suppression: doc={doc_id}")
    
    # Sauvegarder le fichier nettoyé
    try:
        # Préserver le formatage XML original
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)
        
        print()
        print(f"✓ Fichier nettoyé sauvegardé: {output_file}")
        print(f"  - Documents avant: {len(test_textes_dict)}")
        print(f"  - Documents supprimés: {len(doublons)}")
        print(f"  - Documents après: {len(test_textes_dict) - len(doublons)}")
        
    except Exception as e:
        print(f"✗ Erreur lors de la sauvegarde: {e}")


if __name__ == "__main__":
    # Traiter les trois langues: FR, EN, IT
    languages = ['fr', 'en', 'it']
    
    for lang in languages:
        print()
        print(f"Traitement de la langue: {lang.upper()}")
        print("-" * 80)
        
        train_file = f"apprentissage/deft09_parlement_appr_{lang}.xml"
        test_file = f"test/deft09_parlement_test_{lang}.xml"
        output_file = f"test/deft09_parlement_test_{lang}_cleaned.xml"
        
        remove_duplicates(train_file, test_file, output_file)
        print()
    
    print("="*80)
    print("✓ Traitement terminé pour toutes les langues (FR, EN, IT)")
    print("="*80)
