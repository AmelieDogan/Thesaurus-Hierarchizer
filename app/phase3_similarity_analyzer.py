"""
Module d'analyse de similarité lexicale pour la hiérarchisation automatique du thésaurus musical.

Ce module implémente la Phase 3 du cahier des charges pour la détection de relations
hiérarchiques entre termes existants basée sur l'analyse de similarité lexicale :

- Détection par sous-chaînes communes significatives
- Calcul du coefficient de Jaccard sur les n-grammes de mots
- Identification des familles lexicales avec racine commune
- Création de relations hiérarchiques dans les groupes lexicaux

Toutes les relations générées par ce module sont de type ``'existing_parent'`` car
elles opèrent exclusivement sur des termes présents dans le thésaurus avec des URIs.

Attributes:
    logger: Logger pour tracer les opérations du module
    
Example:
    >>> analyzer = LexicalSimilarityAnalyzer(
    ...     min_substring_length=3,
    ...     jaccard_threshold=0.6,
    ...     min_family_size=2,
    ...     max_edit_distance=2
    ... )
    >>> relations = analyzer.detect_lexical_similarities(df, lexical_indexes)
    >>> print(f"Détecté {len(relations)} relations lexicales")
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations

from .logger import get_logger

logger = get_logger(__name__)


class LexicalSimilarityAnalyzer:
    """
    Classe responsable de l'analyse de similarité lexicale avancée pour détecter
    des relations hiérarchiques entre termes existants du thésaurus.
    
    Cette classe implémente la Phase 3 avec plusieurs stratégies complémentaires :
    
    - **Détection par sous-chaînes communes** : Identification de segments textuels
      partagés entre termes d'une longueur minimale significative
    - **Coefficient de Jaccard sur n-grammes** : Mesure de similarité basée sur
      l'intersection et l'union des mots contenus dans les termes
    - **Familles lexicales** : Regroupement de termes partageant une racine commune
    - **Relations hiérarchiques** : Établissement de liens parent-enfant au sein
      des groupes lexicaux identifiés
      
    Toutes les relations générées sont de type ``'existing_parent'`` car cette phase
    travaille exclusivement sur des termes présents dans l'index avec des URIs valides.
    
    Attributes:
        min_substring_length (int): Longueur minimale des sous-chaînes communes
        jaccard_threshold (float): Seuil de similarité Jaccard (conservateur selon CDC)
        min_family_size (int): Taille minimale d'une famille lexicale
        max_edit_distance (int): Distance d'édition maximale pour racine commune
        detected_relations (List[Dict[str, Any]]): Relations détectées par l'analyse
        lexical_families (Dict[str, List[str]]): Familles lexicales identifiées
        similarity_matrix (np.ndarray): Matrice de similarité entre termes
        preflabel_to_uri_map (Dict[str, str]): Mappage prefLabel vers URI
        uri_to_preflabel_map (Dict[str, str]): Mappage URI vers prefLabel
    """
    
    def __init__(self,
                 min_substring_length: int,
                 jaccard_threshold: float,
                 min_family_size: int,
                 max_edit_distance: int):
        """
        Initialise l'analyseur de similarité lexicale avec les paramètres de configuration.
        
        Args:
            min_substring_length: Longueur minimale des sous-chaînes communes à considérer
                comme significatives. Valeur recommandée : 3-4 caractères.
            jaccard_threshold: Seuil de similarité Jaccard au-dessus duquel deux termes
                sont considérés comme similaires. Valeur conservatrice selon CDC : 0.6-0.8.
            min_family_size: Nombre minimum de termes requis pour constituer une famille
                lexicale valide. Valeur recommandée : 2-3 termes.
            max_edit_distance: Distance d'édition maximale autorisée pour considérer
                deux termes comme ayant une racine commune. Valeur recommandée : 1-3.
                
        Note:
            Les paramètres sont calibrés pour un équilibre entre précision et rappel,
            privilégiant la qualité des relations détectées selon une approche conservatrice.
        """
        self.min_substring_length = min_substring_length
        self.jaccard_threshold = jaccard_threshold
        self.min_family_size = min_family_size
        self.max_edit_distance = max_edit_distance
        
        # Statistiques internes d'analyse
        self.detected_relations = []
        self.lexical_families = {}
        self.similarity_matrix = None
        
        # Maps pour les URIs, définies lors de l'appel à detect_lexical_similarities
        self.preflabel_to_uri_map = {}
        self.uri_to_preflabel_map = {}
        
        logger.info(f"Analyseur de similarité lexicale initialisé avec un seuil Jaccard de {jaccard_threshold}")
        
    def find_common_substrings(self, term1: str, term2: str) -> List[str]:
        """
        Trouve les sous-chaînes communes significatives entre deux termes.
        
        Cette méthode recherche toutes les sous-chaînes communes d'au moins
        ``min_substring_length`` caractères, en évitant les redondances
        (une sous-chaîne incluse dans une autre plus longue est ignorée).
        
        Args:
            term1: Premier terme à comparer
            term2: Deuxième terme à comparer
            
        Returns:
            Liste des sous-chaînes communes uniques d'au moins min_substring_length
            caractères, triées par longueur décroissante
            
        Example:
            >>> analyzer = LexicalSimilarityAnalyzer(3, 0.6, 2, 2)
            >>> substrings = analyzer.find_common_substrings("musique", "musicien")
            >>> print(substrings)  # ['musiq', 'musi', etc.]
        """
        if not term1 or not term2:
            return []
            
        # Normalisation pour la comparaison (casse et espaces)
        term1_norm = term1.lower().strip()
        term2_norm = term2.lower().strip()
        
        common_substrings = []
        
        # Recherche exhaustive des sous-chaînes communes
        for i in range(len(term1_norm)):
            for j in range(i + self.min_substring_length, len(term1_norm) + 1):
                substring = term1_norm[i:j]
                if len(substring) >= self.min_substring_length and substring in term2_norm:
                    # Vérifier que ce n'est pas une sous-chaîne d'une déjà trouvée
                    is_new = True
                    for existing in common_substrings:
                        if substring in existing or existing in substring:
                            if len(substring) > len(existing):
                                common_substrings.remove(existing)
                            else:
                                is_new = False
                            break
                    
                    if is_new:
                        common_substrings.append(substring)
        
        return common_substrings
    
    def calculate_jaccard_similarity(self, term1: str, term2: str) -> float:
        """
        Calcule la similarité de Jaccard sur les n-grammes de mots avec bonus
        pour les sous-chaînes communes.
        
        La méthode utilise la formule classique de Jaccard : |A ∩ B| / |A ∪ B|
        où A et B sont les ensembles de mots des deux termes. Un bonus est ajouté
        en fonction de la présence et longueur des sous-chaînes communes.
        
        Args:
            term1: Premier terme à comparer
            term2: Deuxième terme à comparer
            
        Returns:
            Coefficient de Jaccard ajusté (0.0 à 1.0), où 1.0 indique une
            similarité maximale et 0.0 une absence de similarité
            
        Note:
            Le bonus pour sous-chaînes communes est plafonné à 0.2 pour éviter
            la sur-pondération et maintenir la cohérence de l'échelle [0,1].
            
        Example:
            >>> analyzer = LexicalSimilarityAnalyzer(3, 0.6, 2, 2)
            >>> similarity = analyzer.calculate_jaccard_similarity("piano forte", "piano")
            >>> print(f"Similarité: {similarity:.3f}")
        """
        if not term1 or not term2:
            return 0.0
            
        # Tokenisation en mots (alphanumériques uniquement)
        words1 = set(re.findall(r'\b\w+\b', term1.lower()))
        words2 = set(re.findall(r'\b\w+\b', term2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        # Calcul du coefficient de Jaccard classique
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
            
        jaccard_score = len(intersection) / len(union)
        
        # Bonus pour les sous-chaînes communes significatives
        common_substrings = self.find_common_substrings(term1, term2)
        if common_substrings:
            # Bonus proportionnel à la longueur des sous-chaînes communes
            total_common_length = sum(len(s) for s in common_substrings)
            max_term_length = max(len(term1), len(term2))
            substring_bonus = min(0.2, total_common_length / max_term_length)
            jaccard_score = min(1.0, jaccard_score + substring_bonus)
        
        return jaccard_score
    
    def _calculate_edit_distance(self, str1: str, str2: str) -> int:
        """
        Calcule la distance d'édition (Levenshtein) entre deux chaînes.
        
        Utilise l'algorithme de programmation dynamique pour calculer le nombre
        minimum d'opérations (insertion, suppression, substitution) nécessaires
        pour transformer une chaîne en l'autre.
        
        Args:
            str1: Première chaîne de caractères
            str2: Deuxième chaîne de caractères
            
        Returns:
            Distance d'édition (nombre entier >= 0)
            
        Note:
            Optimisation : si str1 est plus courte que str2, elles sont échangées
            pour réduire l'espace mémoire utilisé par l'algorithme.
        """
        if len(str1) < len(str2):
            str1, str2 = str2, str1
            
        if len(str2) == 0:
            return len(str1)
            
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def _extract_root_candidates(self, terms: List[str]) -> List[str]:
        """
        Extrait les candidats racines communes d'une liste de termes par analyse
        morphologique et détection de préfixes.
        
        La méthode combine deux approches :
        1. Détection de préfixes communs entre paires de termes
        2. Analyse morphologique simple avec suppression de suffixes courants
        
        Args:
            terms: Liste des termes à analyser pour l'extraction de racines
            
        Returns:
            Liste des racines candidates uniques, filtrées par longueur minimale
            
        Note:
            Les suffixes supprimés incluent les terminaisons françaises courantes :
            -tion, -ment, -able, -ique, -eur, -euse, -ant, -ent
        """
        if len(terms) < 2:
            return []
            
        root_candidates = []
        
        # Recherche des préfixes communs entre paires
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                term1, term2 = terms[i].lower(), terms[j].lower()
                
                # Calcul du préfixe commun maximal
                common_prefix = ""
                for k in range(min(len(term1), len(term2))):
                    if term1[k] == term2[k]:
                        common_prefix += term1[k]
                    else:
                        break
                
                if len(common_prefix) >= self.min_substring_length:
                    root_candidates.append(common_prefix)
        
        # Analyse morphologique par suppression de suffixes
        for term in terms:
            words = term.lower().split()
            for word in words:
                if len(word) >= self.min_substring_length:
                    # Suppression de suffixes courants français
                    root = word
                    for suffix in ['tion', 'ment', 'able', 'ique', 'eur', 'euse', 'ant', 'ent']:
                        if root.endswith(suffix) and len(root) > len(suffix) + 2:
                            root = root[:-len(suffix)]
                            break
                    
                    if len(root) >= self.min_substring_length:
                        root_candidates.append(root)
        
        return list(set(root_candidates))
    
    def find_lexical_families(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identifie les familles lexicales avec racine commune basées sur la similarité Jaccard.
        
        Cette méthode groupe les termes en familles lexicales en calculant la similarité
        Jaccard pour toutes les paires de termes et en regroupant ceux dépassant le seuil.
        Les familles sont ensuite nettoyées et fusionnées si elles présentent un
        chevauchement significatif.
        
        Args:
            df: DataFrame contenant les données du thésaurus avec colonnes 'preflabel_clean' et 'URI'
            
        Returns:
            Dictionnaire {racine: [termes_de_la_famille]} où chaque famille contient
            au moins min_family_size termes partageant une racine lexicale commune
            
        Side Effects:
            - Met à jour self.lexical_families avec les familles détectées
            - Calcule et stocke self.similarity_matrix pour les termes analysés
            
        Note:
            La fusion des familles se base sur un seuil de chevauchement de 50% pour
            éviter la fragmentation excessive des groupes lexicaux.
        """
        logger.info("Recherche des familles lexicales...")
        
        # Utiliser les prefLabel nettoyés pour la détection
        terms = df['preflabel_clean'].dropna().tolist()
        families = defaultdict(list)
        
        # Créer un mappage de prefLabel_clean à leur index dans le DataFrame original
        preflabel_to_row_index = {label: i for i, label in enumerate(df['preflabel_clean'])}
        
        # Initialisation de la matrice de similarité
        similarity_matrix = np.zeros((len(terms), len(terms)))
        
        # Génération de toutes les paires de termes pour comparaison
        term_pairs = list(combinations(range(len(terms)), 2))

        # Calcul de similarité pour chaque paire avec barre de progression
        for i, j in tqdm(term_pairs, desc="Familles lexicales", unit="paire"):
            term1 = terms[i]
            term2 = terms[j]
            
            similarity = self.calculate_jaccard_similarity(term1, term2)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
            
            # Groupement en famille si similarité élevée
            if similarity >= self.jaccard_threshold:
                # Recherche de la racine commune via sous-chaînes
                common_substrings = self.find_common_substrings(term1, term2)
                if common_substrings:
                    # Utiliser la plus longue sous-chaîne commune comme racine
                    root = max(common_substrings, key=len)
                    families[root].extend([term1, term2])
        
        # Nettoyage des familles : suppression des doublons et filtrage par taille
        cleaned_families = {}
        for root, family_terms in families.items():
            unique_terms = list(set(family_terms))
            if len(unique_terms) >= self.min_family_size:
                cleaned_families[root] = unique_terms
        
        # Fusion des familles avec chevauchement significatif
        final_families = {}
        processed_roots = set()
        
        for root, terms_list in cleaned_families.items():
            if root in processed_roots:
                continue
                
            merged_terms = set(terms_list)
            merged_root = root
            
            # Recherche des familles candidates à la fusion
            for other_root, other_terms in cleaned_families.items():
                if other_root != root and other_root not in processed_roots:
                    # Calcul du taux de chevauchement
                    overlap = len(set(terms_list).intersection(set(other_terms)))
                    min_size = min(len(terms_list), len(other_terms))
                    
                    # Fusion si chevauchement > 50%
                    if overlap > 0 and min_size > 0 and (overlap / min_size) > 0.5:
                        merged_terms.update(other_terms)
                        processed_roots.add(other_root)
                        # Conservation de la racine la plus longue
                        if len(other_root) > len(merged_root):
                            merged_root = other_root
            
            final_families[merged_root] = list(merged_terms)
            processed_roots.add(root)
        
        # Stockage des résultats dans les attributs de classe
        self.lexical_families = final_families
        self.similarity_matrix = similarity_matrix
        
        logger.info(f"Trouvé {len(final_families)} familles lexicales")
        return final_families
    
    def create_lexical_hierarchy(self, df: pd.DataFrame, families: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Crée des relations hiérarchiques parent-enfant dans les familles lexicales détectées.
        
        Cette méthode applique deux stratégies complémentaires pour établir les hiérarchies :
        
        1. **Stratégie générique** : Le terme le plus court devient parent si tous les autres
           termes de la famille le contiennent comme sous-chaîne
        2. **Stratégie par similarité** : Relations basées sur l'inclusion et la longueur
           relative entre paires de termes
        
        Args:
            df: DataFrame contenant les données du thésaurus avec colonnes 'preflabel_clean' et 'URI'
            families: Dictionnaire des familles lexicales {racine: [termes]}
            
        Returns:
            Liste des relations hiérarchiques détectées. Chaque relation contient :
            
            - ``child`` (str): Terme enfant (prefLabel)
            - ``child_uri`` (str): URI du terme enfant
            - ``parent`` (str): Terme parent (prefLabel)  
            - ``parent_uri`` (str): URI du terme parent
            - ``type`` (str): Toujours 'existing_parent' pour cette phase
            - ``relation_category`` (str): Type spécifique ('lexical_family_generic' ou 'lexical_family_similarity')
            - ``confidence`` (float): Niveau de confiance [0.8-0.9]
            - ``source`` (str): 'lexical_similarity'
            - ``details`` (Dict): Métadonnées sur la famille et similarité
            
        Note:
            Toutes les relations générées sont de type 'existing_parent' car cette méthode
            travaille exclusivement sur des termes présents dans le DataFrame avec des URIs.
        """
        logger.info("Création des hiérarchies lexicales...")
        
        relations = []
        
        # Mappage rapide des prefLabel_clean aux URIs
        df_uri_map = df.set_index('preflabel_clean')['URI'].to_dict()
        
        for root, family_terms in families.items():
            if len(family_terms) < 2:
                continue
            
            # Stratégie 1: Terme le plus court comme parent générique
            family_terms_sorted = sorted(family_terms, key=len)
            shortest_term = family_terms_sorted[0]
            
            # Vérification si le terme le plus court peut être un parent générique
            is_generic_parent = True
            for term in family_terms:
                if term != shortest_term and shortest_term.lower() not in term.lower():
                    is_generic_parent = False
                    break
            
            if is_generic_parent:
                # Le terme le plus court devient parent de tous les autres
                parent_uri = df_uri_map.get(shortest_term, self.preflabel_to_uri_map.get(shortest_term, None))
                
                for term in family_terms:
                    if term != shortest_term:
                        child_uri = df_uri_map.get(term, self.preflabel_to_uri_map.get(term, None))
                        
                        relation = {
                            'child': term,
                            'child_uri': child_uri,
                            'parent': shortest_term,
                            'parent_uri': parent_uri,
                            'type': 'existing_parent',  # Nouveau champ normalisé
                            'relation_category': 'lexical_family_generic',  # Ancien type déplacé
                            'confidence': 0.8,
                            'source': 'lexical_similarity',
                            'details': {
                                'family_root': root,
                                'family_size': len(family_terms),
                                'jaccard_similarity': self.calculate_jaccard_similarity(term, shortest_term)
                            }
                        }
                        relations.append(relation)
            else:
                # Stratégie 2: Relations basées sur similarité et inclusion
                for i in range(len(family_terms)):
                    for j in range(i + 1, len(family_terms)):
                        term1, term2 = family_terms[i], family_terms[j]
                        
                        parent, child = None, None
                        
                        # Détermination parent/enfant basée sur longueur et inclusion
                        if len(term1) < len(term2) and term1.lower() in term2.lower():
                            parent, child = term1, term2
                        elif len(term2) < len(term1) and term2.lower() in term1.lower():
                            parent, child = term2, term1
                        else:
                            continue  # Pas de relation hiérarchique claire
                        
                        similarity = self.calculate_jaccard_similarity(child, parent)
                        if similarity >= self.jaccard_threshold:
                            child_uri = df_uri_map.get(child, self.preflabel_to_uri_map.get(child, None))
                            parent_uri = df_uri_map.get(parent, self.preflabel_to_uri_map.get(parent, None))
                            
                            relation = {
                                'child': child,
                                'child_uri': child_uri,
                                'parent': parent,
                                'parent_uri': parent_uri,
                                'type': 'existing_parent',  # Nouveau champ normalisé
                                'relation_category': 'lexical_family_similarity',  # Ancien type déplacé
                                'confidence': min(0.9, similarity),
                                'source': 'lexical_similarity',
                                'details': {
                                    'family_root': root,
                                    'family_size': len(family_terms),
                                    'jaccard_similarity': similarity
                                }
                            }
                            relations.append(relation)
        
        logger.info(f"Créé {len(relations)} relations hiérarchiques lexicales")
        return relations
    
    def _detect_prefix_suffix_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Détecte les relations hiérarchiques basées sur les préfixes et suffixes communs.
        
        Cette méthode groupe les termes par préfixes communs et établit des relations
        hiérarchiques lorsqu'un terme autonome correspond au préfixe d'un groupe.
        
        Args:
            df: DataFrame contenant les données du thésaurus avec colonnes 'preflabel_clean' et 'URI'
            
        Returns:
            Liste des relations hiérarchiques détectées basées sur les préfixes.
            Chaque relation suit la même structure que create_lexical_hierarchy()
            avec relation_category = 'lexical_prefix' et type = 'existing_parent'.
            
        Note:
            Le seuil de similarité est abaissé à 0.5 pour les relations de préfixe
            car la structure morphologique est déjà un indicateur fort de relation.
        """
        terms = df['preflabel_clean'].dropna().tolist()
        relations = []
        
        # Mappage rapide des prefLabel_clean aux URIs
        df_uri_map = df.set_index('preflabel_clean')['URI'].to_dict()
        
        # Groupement par préfixes communs (premier mot)
        prefix_groups = defaultdict(list)
        
        for term in terms:
            words = term.lower().split()
            if len(words) > 1:
                # Premier mot comme préfixe potentiel
                prefix = words[0]
                if len(prefix) >= self.min_substring_length:
                    prefix_groups[prefix].append(term)
        
        # Création de relations pour les groupes de préfixes valides
        for prefix, group_terms in prefix_groups.items():
            if len(group_terms) >= self.min_family_size:
                # Recherche d'un terme autonome correspondant au préfixe
                prefix_term = None
                for term in terms:
                    if term.lower().strip() == prefix:
                        prefix_term = term
                        break
                
                if prefix_term:
                    parent_uri = df_uri_map.get(prefix_term, self.preflabel_to_uri_map.get(prefix_term, None))
                    
                    # Création des relations hiérarchiques
                    for term in group_terms:
                        if term != prefix_term:
                            similarity = self.calculate_jaccard_similarity(term, prefix_term)
                            if similarity >= 0.5:  # Seuil plus bas pour les préfixes
                                child_uri = df_uri_map.get(term, self.preflabel_to_uri_map.get(term, None))
                                
                                relation = {
                                    'child': term,
                                    'child_uri': child_uri,
                                    'parent': prefix_term,
                                    'parent_uri': parent_uri,
                                    'type': 'existing_parent',  # Nouveau champ normalisé
                                    'relation_category': 'lexical_prefix',  # Ancien type déplacé
                                    'confidence': min(0.8, similarity + 0.2),
                                    'source': 'lexical_similarity',
                                    'details': {
                                        'prefix': prefix,
                                        'group_size': len(group_terms),
                                        'jaccard_similarity': similarity
                                    }
                                }
                                relations.append(relation)
        
        return relations
    
    def detect_lexical_similarities(self, df: pd.DataFrame, lexical_indexes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Détecte toutes les similarités lexicales avancées et génère les relations hiérarchiques.
        
        Cette méthode orchestre l'ensemble du processus d'analyse de similarité lexicale :
        
        1. **Initialisation** des mappages URI/prefLabel à partir des index lexicaux
        2. **Détection des familles lexicales** basée sur la similarité Jaccard
        3. **Création de hiérarchies** au sein des familles détectées
        4. **Détection des relations préfixe/suffixe** complémentaires
        5. **Déduplication et consolidation** des relations trouvées
        
        Args:
            df: DataFrame contenant les données du thésaurus avec colonnes obligatoires :
                - 'preflabel_clean' : Labels nettoyés pour l'analyse
                - 'URI' : Identifiants uniques des concepts
            lexical_indexes: Dictionnaire des index lexicaux contenant :
                - 'preflabel_to_uri' : Mappage prefLabel vers URI
                - 'uri_to_preflabel' : Mappage URI vers prefLabel
                
        Returns:
            Liste consolidée de toutes les relations détectées par similarité lexicale.
            Chaque relation est un dictionnaire avec la structure suivante :
            
            - ``child`` (str): Terme enfant (prefLabel)
            - ``child_uri`` (str): URI du terme enfant  
            - ``parent`` (str): Terme parent (prefLabel)
            - ``parent_uri`` (str): URI du terme parent
            - ``type`` (str): Toujours 'existing_parent' 
            - ``relation_category`` (str): Catégorie spécifique de relation
            - ``confidence`` (float): Niveau de confiance [0.5-0.9]
            - ``source`` (str): 'lexical_similarity'
            - ``details`` (Dict): Métadonnées spécifiques à la méthode de détection
            
        Side Effects:
            - Met à jour self.detected_relations avec toutes les relations trouvées
            - Met à jour self.preflabel_to_uri_map et self.uri_to_preflabel_map
            
        Note:
            La déduplication privilégie les relations avec la meilleure confiance
            en cas de doublons basés sur la paire (child_uri, parent_uri).
            
        Example:
            >>> analyzer = LexicalSimilarityAnalyzer(3, 0.6, 2, 2)
            >>> relations = analyzer.detect_lexical_similarities(df, indexes)
            >>> for rel in relations[:3]:
            ...     print(f"{rel['child']} -> {rel['parent']} ({rel['confidence']:.2f})")
        """
        logger.info("Début de l'analyse de similarité lexicale...")
        
        # Initialisation des mappages URI/prefLabel depuis les index lexicaux
        self.preflabel_to_uri_map = lexical_indexes.get('preflabel_to_uri', {})
        self.uri_to_preflabel_map = lexical_indexes.get('uri_to_preflabel', {})
        
        all_relations = []
        
        # 1. Détection des familles lexicales et création des hiérarchies
        families = self.find_lexical_families(df)
        family_relations = self.create_lexical_hierarchy(df, families)
        all_relations.extend(family_relations)
        
        # 2. Détection des relations préfixe/suffixe complémentaires
        prefix_relations = self._detect_prefix_suffix_relations(df)
        all_relations.extend(prefix_relations)
        
        # 3. Déduplication et consolidation des relations
        unique_relations = []
        seen_pairs = set()
        
        for relation in all_relations:
            # Utilisation des URIs pour identifier les paires uniques
            pair = (relation['child_uri'], relation['parent_uri'])
            if pair not in seen_pairs:
                unique_relations.append(relation)
                seen_pairs.add(pair)
            else:
                # Conservation de la relation avec la meilleure confiance
                for i, existing in enumerate(unique_relations):
                    if (existing['child_uri'], existing['parent_uri']) == pair:
                        if relation['confidence'] > existing['confidence']:
                            unique_relations[i] = relation
                        break
        
        # Stockage des relations finales
        self.detected_relations = unique_relations
        
        logger.info(f"Détecté {len(unique_relations)} relations par similarité lexicale")
        return unique_relations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées de l'analyse de similarité lexicale.
        
        Compile des métriques complètes sur les relations détectées, incluant
        les distributions par type, niveau de confiance, et méthode de détection.
        
        Returns:
            Dictionnaire contenant les statistiques suivantes :
            
            - ``total_relations`` (int): Nombre total de relations détectées
            - ``lexical_families_count`` (int): Nombre de familles lexicales identifiées
            - ``relations_by_category`` (Dict[str, int]): Distribution par relation_category
            - ``confidence_distribution`` (Dict[str, int]): Distribution par tranches de confiance
            - ``method_distribution`` (Dict[str, int]): Distribution par méthode source
            
        Note:
            Les tranches de confiance sont formatées comme "0.7-0.8" pour faciliter
            l'interprétation des résultats d'analyse.
            
        Example:
            >>> stats = analyzer.get_statistics()
            >>> print(f"Relations détectées: {stats['total_relations']}")
            >>> print(f"Familles lexicales: {stats['lexical_families_count']}")
        """
        stats = {
            'total_relations': len(self.detected_relations),
            'lexical_families_count': len(self.lexical_families),
            'relations_by_category': defaultdict(int),  # Nouveau nom pour l'ancien 'relations_by_type'
            'confidence_distribution': defaultdict(int),
            'method_distribution': defaultdict(int)
        }
        
        for relation in self.detected_relations:
            # Utilisation du nouveau champ relation_category
            stats['relations_by_category'][relation['relation_category']] += 1
            stats['method_distribution'][relation['source']] += 1
            
            # Distribution par tranches de confiance de 0.1
            confidence_range = f"{int(relation['confidence'] * 10) / 10:.1f}-{int(relation['confidence'] * 10) / 10 + 0.1:.1f}"
            stats['confidence_distribution'][confidence_range] += 1
        
        return dict(stats)
    
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Retourne la matrice de similarité Jaccard calculée entre tous les termes.
        
        La matrice est carrée et symétrique, où element[i,j] représente la similarité
        Jaccard entre les termes aux positions i et j dans la liste des termes analysés.
        
        Returns:
            Matrice de similarité numpy de dimensions (n_terms, n_terms) où
            les valeurs sont comprises entre 0.0 (aucune similarité) et 1.0 (identiques)
            
        Note:
            La matrice est calculée lors de l'appel à find_lexical_families() et
            peut être None si cette méthode n'a pas encore été exécutée.
            
        Example:
            >>> matrix = analyzer.get_similarity_matrix()
            >>> if matrix is not None:
            ...     print(f"Taille de la matrice: {matrix.shape}")
            ...     print(f"Similarité maximale: {matrix.max():.3f}")
        """
        return self.similarity_matrix
    
    def export_families_to_dataframe(self) -> pd.DataFrame:
        """
        Exporte les familles lexicales identifiées vers un DataFrame pour analyse.
        
        Crée une représentation tabulaire des familles lexicales où chaque ligne
        correspond à un terme membre d'une famille, avec ses métadonnées associées.
        
        Returns:
            DataFrame avec les colonnes suivantes :
            
            - ``family_root`` (str): Racine lexicale de la famille
            - ``term`` (str): Terme membre de la famille (prefLabel)
            - ``family_size`` (int): Nombre total de termes dans la famille
            - ``term_uri`` (str): URI du terme membre
            
        Note:
            Utile pour l'analyse manuelle des familles détectées et la validation
            de la qualité du regroupement lexical.
            
        Example:
            >>> families_df = analyzer.export_families_to_dataframe()
            >>> print(f"Familles exportées: {len(families_df)} termes")
            >>> print(families_df.groupby('family_root').size().head())
        """
        family_data = []
        
        for root, terms in self.lexical_families.items():
            for term in terms:
                family_data.append({
                    'family_root': root,
                    'term': term,
                    'family_size': len(terms),
                    'term_uri': self.preflabel_to_uri_map.get(term, None)
                })
        
        return pd.DataFrame(family_data)
    
    def export_relations_to_dataframe(self) -> pd.DataFrame:
        """
        Exporte toutes les relations détectées vers un DataFrame pour analyse et sauvegarde.
        
        Convertit la liste des relations en format tabulaire pour faciliter
        l'analyse statistique, la validation manuelle et l'export vers d'autres formats.
        
        Returns:
            DataFrame contenant toutes les colonnes des relations détectées :
            child, child_uri, parent, parent_uri, type, relation_category,
            confidence, source, et details (sérialisé)
            
        Note:
            Le champ 'details' est conservé tel quel (dictionnaire) dans le DataFrame.
            Pour un export vers CSV, une sérialisation supplémentaire peut être nécessaire.
            
        Example:
            >>> relations_df = analyzer.export_relations_to_dataframe()
            >>> print(f"Relations exportées: {len(relations_df)}")
            >>> print(relations_df['relation_category'].value_counts())
        """
        return pd.DataFrame(self.detected_relations)
    
    def get_relations_by_confidence(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Filtre et retourne les relations selon un seuil de confiance minimum.
        
        Permet de sélectionner uniquement les relations dépassant un niveau
        de confiance spécifique pour des analyses ciblées ou des exports qualifiés.
        
        Args:
            min_confidence: Seuil de confiance minimum (0.0 à 1.0).
                Valeur par défaut 0.0 retourne toutes les relations.
                
        Returns:
            Liste des relations ayant une confiance >= min_confidence,
            conservant la structure originale des dictionnaires de relation
            
        Example:
            >>> # Relations très fiables uniquement
            >>> high_conf_relations = analyzer.get_relations_by_confidence(0.8)
            >>> print(f"Relations haute confiance: {len(high_conf_relations)}")
            >>> 
            >>> # Relations moyennes et hautes
            >>> mid_conf_relations = analyzer.get_relations_by_confidence(0.6)
        """
        return [r for r in self.detected_relations if r['confidence'] >= min_confidence]
    
    def validate_relation_quality(self, sample_size: int = 20) -> Dict[str, Any]:
        """
        Génère un échantillon stratifié de relations pour validation manuelle de la qualité.
        
        Sélectionne un échantillon représentatif des différents types de relations
        détectées, privilégiant celles avec la meilleure confiance pour faciliter
        l'évaluation de la performance de l'algorithme.
        
        Args:
            sample_size: Taille cible de l'échantillon à générer.
                La taille réelle peut être inférieure si peu de relations sont disponibles.
                
        Returns:
            Dictionnaire contenant :
            
            - ``sample`` (List[Dict]): Relations échantillonnées pour validation
            - ``metrics`` (Dict): Métriques de l'échantillon incluant :
                - ``sample_size`` (int): Taille effective de l'échantillon
                - ``avg_confidence`` (float): Confiance moyenne de l'échantillon
                - ``confidence_std`` (float): Écart-type de la confiance
                - ``types_represented`` (List[str]): Categories de relations représentées
                
        Note:
            L'échantillonnage est stratifié par relation_category pour assurer
            une représentativité équilibrée des différents types de détection.
            
        Example:
            >>> validation = analyzer.validate_relation_quality(10)
            >>> print(f"Échantillon de {validation['metrics']['sample_size']} relations")
            >>> print(f"Confiance moyenne: {validation['metrics']['avg_confidence']:.3f}")
            >>> for rel in validation['sample'][:3]:
            ...     print(f"  {rel['child']} -> {rel['parent']} ({rel['relation_category']})")
        """
        if len(self.detected_relations) == 0:
            return {'sample': [], 'metrics': {}}
        
        # Échantillonnage stratifié par catégorie de relation
        sample_relations = []
        relations_by_category = defaultdict(list)
        
        # Groupement par relation_category (nouveau champ)
        for relation in self.detected_relations:
            relations_by_category[relation['relation_category']].append(relation)
        
        # Échantillonnage proportionnel de chaque catégorie
        for rel_category, relations in relations_by_category.items():
            category_sample_size = min(len(relations), max(1, sample_size // len(relations_by_category)))
            # Sélection des relations avec la meilleure confiance
            relations_sorted = sorted(relations, key=lambda x: x['confidence'], reverse=True)
            sample_relations.extend(relations_sorted[:category_sample_size])
        
        # Calcul des métriques de l'échantillon
        sample_metrics = {
            'sample_size': len(sample_relations),
            'avg_confidence': np.mean([r['confidence'] for r in sample_relations]),
            'confidence_std': np.std([r['confidence'] for r in sample_relations]),
            'types_represented': list(set(r['relation_category'] for r in sample_relations))
        }
        
        return {
            'sample': sample_relations,
            'metrics': sample_metrics
        }