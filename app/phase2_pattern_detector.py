"""
Module de détection des patterns lexicaux pour la hiérarchisation du thésaurus.
Ce module implémente la Phase 2 : détection des relations par patterns lexicaux.
"""

import pandas as pd
import re
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter

from .utils import is_valid_word, normalize_text
from .config import get_default_stop_words

from .logger import get_logger

logger = get_logger(__name__)

class LexicalPatternDetector:
    """
    Classe responsable de la détection des patterns lexicaux dans le thésaurus.
    
    Cette classe implémente la Phase 2 avec :
    - Pattern principal d'inclusion de mots
    - Patterns spécialisés musicaux (X de Y, X en Y, etc.)
    - Gestion des mots-outils
    - Scoring des relations détectées
    - Détection de parents potentiels non-existants
    """
    
    def __init__(self, min_frequency_for_candidate: int):
        """
        Initialise le détecteur de patterns lexicaux.
        
        Args:
            min_frequency_for_candidate: Fréquence minimale pour proposer un mot comme parent candidat
        """
        self.stop_words = get_default_stop_words()
        self.relations_found = []
        self.statistics = defaultdict(int)
        self.min_frequency_for_candidate = min_frequency_for_candidate
        self.word_frequency = Counter()
        self.candidate_parents = set()
        
        # Ces maps seront définies lors de l'appel à detect_all_patterns
        self.preflabel_to_uri_map = {}
        self.uri_to_preflabel_map = {}
        
        logger.info(f"Détecteur de patterns initialisé avec {len(self.stop_words)} mots-outils")
        logger.info(f"Fréquence minimale pour candidats parents : {min_frequency_for_candidate}")
    
    def decompose_multiword_term(self, term: str) -> List[str]:
        """
        Décompose un terme multi-mots en mots individuels significatifs.
        
        Args:
            term: Terme à décomposer
            
        Returns:
            Liste des mots significatifs (sans mots-outils)
        """
        if not term or pd.isna(term):
            return []
        
        # Nettoyage et séparation des mots
        # Remplacer les tirets par des espaces pour la décomposition
        term_clean = re.sub(r'[-_]', ' ', term.lower())
        words = re.findall(r'\b\w+\b', term_clean)
        
        # Filtrer les mots valides
        valid_words = [word for word in words if is_valid_word(word)]
        
        return valid_words
    
    def analyze_word_frequency(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyse la fréquence des mots dans tous les termes pour identifier les candidats parents.
        
        Args:
            df: DataFrame contenant les données du thésaurus
            
        Returns:
            Dictionnaire {mot: fréquence}
        """
        logger.info("Analyse de la fréquence des mots dans le thésaurus")
        
        word_counter = Counter()
        processed_terms = 0
        
        for _, row in df.iterrows():
            preflabel = row['preflabel_clean']
            normalized = row['preflabel_normalized']
            
            if not preflabel or not normalized:
                continue
            
            words = self.decompose_multiword_term(normalized)
            if len(words) > 1:  # Seulement les termes multi-mots
                word_counter.update(words)
                processed_terms += 1
        
        self.word_frequency = word_counter
        
        # Identifier les candidats parents potentiels
        frequent_words = {word: freq for word, freq in word_counter.items() 
                         if freq >= self.min_frequency_for_candidate}
        
        logger.info("Analyse de fréquence terminée :")
        logger.info(f"  - {processed_terms} termes multi-mots traités")
        logger.info(f"  - {len(word_counter)} mots uniques trouvés")
        logger.info(f"  - {len(frequent_words)} mots fréquents (>= {self.min_frequency_for_candidate})")
        
        if frequent_words:
            top_frequent = word_counter.most_common(10)
            logger.info(f"  - Top 10 mots fréquents : {top_frequent}")
        
        return frequent_words
    
    def detect_word_inclusion_patterns(self, 
                                     df: pd.DataFrame) -> List[Dict[str, Any]]: # Removed word_index
        """
        Détecte les patterns d'inclusion de mots (règle fondamentale).
        Maintenant détecte aussi les parents potentiels non-existants.
        
        Args:
            df: DataFrame contenant les données du thésaurus
            
        Returns:
            Liste des relations détectées
        """
        logger.info("Détection des patterns d'inclusion de mots")
        
        relations = []
        
        # Analyser la fréquence des mots
        frequent_words = self.analyze_word_frequency(df)
        
        # Index des termes existants pour recherche rapide
        preflabel_to_normalized = {row['preflabel_clean']: row['preflabel_normalized'] for _, row in df.iterrows() if pd.notna(row['preflabel_clean']) and pd.notna(row['preflabel_normalized'])}
        existing_normalized_terms = set(preflabel_to_normalized.values())
        
        logger.info(f"Index des termes existants (normalisés) : {len(existing_normalized_terms)} termes")
        
        # Compteurs pour les statistiques détaillées
        existing_parent_matches = 0
        candidate_parent_matches = 0
        
        # Pour chaque terme multi-mots
        for _, row in df.iterrows():
            preflabel = row['preflabel_clean']
            normalized = row['preflabel_normalized']
            child_uri = row['URI'] # Récupérer l'URI de l'enfant
            
            if not preflabel or not normalized or pd.isna(child_uri):
                continue
            
            words = self.decompose_multiword_term(normalized)
            
            # Si c'est un terme multi-mots
            if len(words) > 1:
                # Vérifier chaque mot
                for word in words:
                    parent_preflabel = None
                    parent_uri = None
                    relation_type = None
                    
                    # 1. Vérifier si le mot existe comme terme autonome
                    if word in existing_normalized_terms:
                        # Trouver le prefLabel correspondant et son URI
                        for p_label, p_norm in preflabel_to_normalized.items():
                            if p_norm == word:
                                parent_preflabel = p_label
                                parent_uri = self.preflabel_to_uri_map.get(normalize_text(parent_preflabel))
                                relation_type = 'existing_parent'
                                existing_parent_matches += 1
                                break
                    
                    # 2. Si pas trouvé ET que le mot est fréquent, le proposer comme candidat
                    if not parent_preflabel and word in frequent_words:
                        parent_preflabel = word  # Utiliser le mot lui-même comme parent candidat
                        parent_uri = None # L'URI sera générée en Phase 5 pour les nouveaux termes
                        relation_type = 'candidate_parent'
                        candidate_parent_matches += 1
                        self.candidate_parents.add(word)
                    
                    # Créer la relation si un parent a été trouvé
                    if parent_preflabel:
                        relation = {
                            'child': preflabel,
                            'child_uri': child_uri,
                            'parent': parent_preflabel,
                            'parent_uri': parent_uri,
                            'relation_category': 'word_inclusion',
                            'pattern_detail': f'"{preflabel}" contient "{word}"',
                            'confidence': 1.0 if relation_type == 'existing_parent' else 0.8,
                            'source_word': word,
                            'child_words': words,
                            'type': relation_type,
                            'source': 'pattern_inclusion',
                            'word_frequency': frequent_words.get(word, self.word_frequency.get(word, 0))
                        }
                        relations.append(relation)
        
        # Mise à jour des statistiques
        self.statistics['word_inclusion_relations'] = len(relations)
        self.statistics['existing_parent_matches'] = existing_parent_matches
        self.statistics['candidate_parent_matches'] = candidate_parent_matches
        self.statistics['candidate_parents_found'] = len(self.candidate_parents)
        
        logger.info("Détection d'inclusion terminée :")
        logger.info(f"  - {len(relations)} relations trouvées")
        logger.info(f"  - {existing_parent_matches} avec parents existants")
        logger.info(f"  - {candidate_parent_matches} avec parents candidats")
        logger.info(f"  - {len(self.candidate_parents)} parents candidats uniques")
        
        if self.candidate_parents:
            logger.info(f"  - Exemples de parents candidats : {list(self.candidate_parents)[:10]}")
        
        return relations
    
    def detect_pattern_with_candidates(self, 
                                     df: pd.DataFrame, 
                                     pattern: re.Pattern,
                                     pattern_name: str,
                                     confidence: float = 0.9) -> List[Dict[str, Any]]:
        """
        Détecte un pattern spécifique en incluant les parents candidats.
        
        Args:
            df: DataFrame contenant les données du thésaurus
            pattern: Pattern regex compilé
            pattern_name: Nom du pattern pour les logs
            confidence: Score de confiance pour ce pattern
            
        Returns:
            Liste des relations détectées
        """
        logger.info(f"Détection du pattern '{pattern_name}'")
        
        relations = []
        
        # Index des termes existants pour vérification
        preflabel_to_normalized = {row['preflabel_clean']: row['preflabel_normalized'] for _, row in df.iterrows() if pd.notna(row['preflabel_clean']) and pd.notna(row['preflabel_normalized'])}
        existing_normalized_terms = set(preflabel_to_normalized.values())
        
        existing_matches = 0
        candidate_matches = 0
        
        for _, row in df.iterrows():
            preflabel = row['preflabel_clean']
            normalized = row['preflabel_normalized']
            child_uri = row['URI'] # Récupérer l'URI de l'enfant
            
            if not preflabel or not normalized or pd.isna(child_uri):
                continue
            
            match = pattern.match(normalized)
            if match:
                base_term = match.group(1).strip()
                complement = match.group(2).strip()
                
                parent_preflabel = None
                parent_uri = None
                relation_type = None
                
                # 1. Vérifier si le terme de base existe
                if base_term in existing_normalized_terms:
                    # Trouver le prefLabel correspondant et son URI
                    for p_label, p_norm in preflabel_to_normalized.items():
                        if p_norm == base_term:
                            parent_preflabel = p_label
                            parent_uri = self.preflabel_to_uri_map.get(normalize_text(parent_preflabel))
                            relation_type = 'existing_parent'
                            existing_matches += 1
                            break
                
                # 2. Si pas trouvé ET que le terme de base est fréquent, le proposer comme candidat
                elif base_term in self.word_frequency and self.word_frequency[base_term] >= self.min_frequency_for_candidate:
                    parent_preflabel = base_term
                    parent_uri = None # L'URI sera générée en Phase 5 pour les nouveaux termes
                    relation_type = 'candidate_parent'
                    candidate_matches += 1
                    self.candidate_parents.add(base_term)
                
                # Créer la relation si un parent a été trouvé
                if parent_preflabel:
                    relation = {
                        'child': preflabel,
                        'child_uri': child_uri,
                        'parent': parent_preflabel,
                        'parent_uri': parent_uri,
                        'relation_category': pattern_name,
                        'pattern_detail': f'"{preflabel}" suit le pattern "{pattern_name}"',
                        'confidence': confidence if relation_type == 'existing_parent' else confidence * 0.8,
                        'base_term': base_term,
                        'complement': complement,
                        'type': relation_type,
                        'source': 'pattern_specialized',
                        'word_frequency': self.word_frequency.get(base_term, 0)
                    }
                    relations.append(relation)
        
        # Mise à jour des statistiques
        self.statistics[f'{pattern_name}_relations'] = len(relations)
        self.statistics[f'{pattern_name}_existing_matches'] = existing_matches
        self.statistics[f'{pattern_name}_candidate_matches'] = candidate_matches
        logger.info(f"Pattern '{pattern_name}' terminé :")
        logger.info(f"  - {len(relations)} relations trouvées")
        logger.info(f"  - {existing_matches} avec parents existants")
        logger.info(f"  - {candidate_matches} avec parents candidats")
        return relations
    
    def detect_de_pattern(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Détecte le pattern 'X de Y' → X enfant de Y (ou X enfant de X-base).
        """
        pattern = re.compile(r'^(.+?)\s+de\s+(.+)$', re.IGNORECASE)
        return self.detect_pattern_with_candidates(df, pattern, 'de_pattern', 0.9)

    def detect_en_pattern(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Détecte le pattern 'X en Y' → X enfant de Y (ou X enfant de X-base).
        """
        pattern = re.compile(r'^(.+?)\s+en\s+(.+)$', re.IGNORECASE)
        return self.detect_pattern_with_candidates(df, pattern, 'en_pattern', 0.9)

    def detect_pour_pattern(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Détecte le pattern 'X pour Y' → X enfant de Y (ou X enfant de X-base).
        """
        pattern = re.compile(r'^(.+?)\s+pour\s+(.+)$', re.IGNORECASE)
        return self.detect_pattern_with_candidates(df, pattern, 'pour_pattern', 0.9)

    def detect_adjectival_pattern(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Détecte le pattern adjectival 'X adjectif' → X enfant de X-base.
        """
        logger.info("Détection du pattern adjectival")
        relations = []
        # Liste d'adjectifs musicaux courants (à enrichir selon le domaine)
        musical_adjectives = [
            "musical", "musicale", "musicaux", "musicales", "sonore", "sonores",
            "vocal", "vocale", "vocaux", "vocales", "instrumental", "instrumentale",
            "instrumentaux", "instrumentales", "choral", "chorale", "choraux",
            "chorales", "symphonique", "symphoniques", "lyrique", "lyriques",
            "opératique", "opératiques", "sacré", "sacrée", "sacrés", "sacrées",
            "profane", "profanes", "traditionnel", "traditionnelle", "traditionnels",
            "traditionnelles", "populaire", "populaires", "classique", "classiques",
            "contemporain", "contemporaine", "contemporains", "contemporaines",
            "ancien", "ancienne", "anciens", "anciennes", "moderne", "modernes",
            "majeur", "majeure", "majeurs", "majeures", "mineur", "mineure",
            "mineurs", "mineures", "folklorique", "folkloriques", "jazz", "rock",
            "électronique", "électroniques", "numérique", "numériques", "digitale",
            "digitales", "digital", "digitals", "acoustique", "acoustiques"
        ]
        
        # Compiler les adjectifs pour une recherche rapide
        adj_pattern = re.compile(r'\b(' + '|'.join(re.escape(adj) for adj in musical_adjectives) + r')\b', re.IGNORECASE)

        preflabel_to_normalized = {row['preflabel_clean']: row['preflabel_normalized'] for _, row in df.iterrows() if pd.notna(row['preflabel_clean']) and pd.notna(row['preflabel_normalized'])}
        existing_normalized_terms = set(preflabel_to_normalized.values())
        
        existing_matches = 0
        candidate_matches = 0

        for _, row in df.iterrows():
            preflabel = row['preflabel_clean']
            normalized = row['preflabel_normalized']
            child_uri = row['URI'] # Récupérer l'URI de l'enfant

            if not preflabel or not normalized or pd.isna(child_uri):
                continue
            
            # Pattern: "NOM ADJECTIF" -> NOM
            match_adj = adj_pattern.search(normalized)
            if match_adj:
                adjective = match_adj.group(1)
                base_term_raw = normalized.replace(adjective, '').strip()
                base_term = normalize_text(base_term_raw) # Normaliser pour la recherche

                if not base_term: # Éviter les cas où seul l'adjectif est présent
                    continue

                parent_preflabel = None
                parent_uri = None
                relation_type = None

                # 1. Vérifier si le terme de base existe
                if base_term in existing_normalized_terms:
                    for p_label, p_norm in preflabel_to_normalized.items():
                        if p_norm == base_term:
                            parent_preflabel = p_label
                            parent_uri = self.preflabel_to_uri_map.get(normalize_text(parent_preflabel))
                            relation_type = 'existing_parent'
                            existing_matches += 1
                            break
                # 2. Si pas trouvé ET que le terme de base est fréquent, le proposer comme candidat
                elif base_term in self.word_frequency and self.word_frequency[base_term] >= self.min_frequency_for_candidate:
                    parent_preflabel = base_term
                    parent_uri = None # L'URI sera générée en Phase 5
                    relation_type = 'candidate_parent'
                    candidate_matches += 1
                    self.candidate_parents.add(base_term)
                
                if parent_preflabel:
                    relations.append({
                        'child': preflabel,
                        'child_uri': child_uri,
                        'parent': parent_preflabel,
                        'parent_uri': parent_uri,
                        'relation_category': 'adjectival_pattern',
                        'pattern_detail': f'"{preflabel}" est une variation de "{base_term}"',
                        'confidence': 0.8 if relation_type == 'existing_parent' else 0.6,
                        'base_term': base_term,
                        'adjective': adjective,
                        'type': relation_type,
                        'source': 'pattern_specialized',
                        'word_frequency': self.word_frequency.get(base_term, 0)
                    })
        
        self.statistics['adjectival_relations'] = len(relations)
        self.statistics['adjectival_existing_matches'] = existing_matches
        self.statistics['adjectival_candidate_matches'] = candidate_matches
        logger.info(f"Pattern adjectival terminé :")
        logger.info(f"  - {len(relations)} relations trouvées")
        logger.info(f"  - {existing_matches} avec parents existants")
        logger.info(f"  - {candidate_matches} avec parents candidats")
        return relations


    def detect_all_patterns(self, df: pd.DataFrame, lexical_indexes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Exécute toutes les détections de patterns et agrège les résultats.
        
        Args:
            df: DataFrame contenant les données du thésaurus (avec 'URI' column)
            lexical_indexes: Dictionnaire des index lexicaux (incluant uri_to_preflabel et preflabel_to_uri)
            
        Returns:
            Liste consolidée de toutes les relations détectées par patterns
        """
        logger.info("Détection de tous les patterns lexicaux.")
        
        self.preflabel_to_uri_map = lexical_indexes.get('preflabel_to_uri', {})
        self.uri_to_preflabel_map = lexical_indexes.get('uri_to_preflabel', {})

        self.relations_found = []
        self.statistics = defaultdict(int) # Réinitialiser les stats pour cet appel
        
        # Assurez-vous que le DataFrame a bien la colonne 'URI'
        if 'URI' not in df.columns:
            logger.error("La colonne 'URI' est manquante dans le DataFrame. Impossible de générer les URIs des relations.")
            return []

        # 1. Patterns d'inclusion de mots (Règle fondamentale)
        word_inclusion_relations = self.detect_word_inclusion_patterns(df)
        self.relations_found.extend(word_inclusion_relations)

        # 2. Patterns spécialisés (avec gestion des candidats)
        self.relations_found.extend(self.detect_de_pattern(df))
        self.relations_found.extend(self.detect_en_pattern(df))
        self.relations_found.extend(self.detect_pour_pattern(df))
        self.relations_found.extend(self.detect_adjectival_pattern(df))

        logger.info(f"Total des relations détectées par patterns : {len(self.relations_found)}")
        
        return self.relations_found

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la détection de patterns.
        
        Returns:
            Dictionnaire des statistiques
        """
        return dict(self.statistics)

    def get_candidate_parents(self) -> Set[str]:
        """
        Retourne les parents candidats uniques détectés.
        
        Returns:
            Set de strings des parents candidats
        """
        return self.candidate_parents
    
    def get_word_frequency(self) -> Counter:
        """
        Retourne la fréquence des mots analysée.
        
        Returns:
            Counter des mots et de leurs fréquences
        """
        return self.word_frequency