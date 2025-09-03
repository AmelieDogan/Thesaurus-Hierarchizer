"""
Module de prétraitement des données de thésaurus - Phase 1.

Ce module contient la classe principale pour le nettoyage et la préparation
des données SKOS en vue de la hiérarchisation automatique.
"""

import pandas as pd
import re
import uuid
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter

from .utils import normalize_text

from .logger import get_logger

logger = get_logger(__name__)


class ThesaurusDataProcessor:
    """
    Processeur de données pour le nettoyage et la préparation des données de thésaurus.
    
    Cette classe implémente toutes les étapes de la Phase 1 du pipeline de hiérarchisation :
    
    * Nettoyage des définitions et extraction des sources
    * Extraction des relations skos:broader depuis les prefLabel
    * Normalisation des labels
    * Explosion des altLabel multiples
    * Création des index lexicaux avec URIs
    * Extraction des relations skos:broader existantes
    * Analyse des relations depuis broader_from_preflabel
    
    Attributes:
        tsv_file_path (Path): Chemin vers le fichier TSV source
        raw_data (pd.DataFrame): Données brutes chargées
        processed_data (pd.DataFrame): Données après traitement
        lexical_indexes (Dict[str, Any]): Index lexicaux pour la recherche rapide
        existing_broader_relations (List[Dict]): Relations skos:broader existantes
        preflabel_broader_relations (List[Dict]): Relations extraites des prefLabel
        uri_base (str): Base URI pour la génération des identifiants
        min_frequency_for_candidate (int): Seuil de fréquence pour les parents candidats
        word_frequency (Counter): Compteur de fréquence des termes
        candidate_parents (set): Ensemble des parents candidats détectés
    
    Example:
        >>> processor = ThesaurusDataProcessor(
        ...     "data/thesaurus.tsv",
        ...     "https://example.org/thesaurus/",
        ...     min_frequency_for_candidate=3
        ... )
        >>> df, indexes, relations = processor.preprocess_data()
        >>> print(f"Traité {len(df)} termes avec {len(relations)} relations")
        
    .. versionadded:: 1.0
        Support initial des thésaurus SKOS
        
    .. versionchanged:: 1.1
        Ajout de l'analyse des relations depuis broader_from_preflabel
    """
    
    def __init__(self, tsv_file_path: str, uri_base: str, min_frequency_for_candidate: int = 2):
        """
        Initialise le processeur de données du thésaurus.
        
        Args:
            tsv_file_path: Chemin vers le fichier TSV contenant les données
            uri_base: Base URI pour la génération des URIs des termes
            min_frequency_for_candidate: Fréquence minimale pour proposer un parent candidat
        
        Raises:
            FileNotFoundError: Si le fichier TSV n'existe pas
        """
        self.tsv_file_path = Path(tsv_file_path)
        if not self.tsv_file_path.exists():
            raise FileNotFoundError(f"Le fichier {tsv_file_path} n'existe pas")
        
        self.raw_data = None
        self.processed_data = None
        self.lexical_indexes = {}
        self.existing_broader_relations = []
        self.preflabel_broader_relations = []  # Nouveau: relations depuis broader_from_preflabel
        self.uri_base = uri_base
        self.min_frequency_for_candidate = min_frequency_for_candidate
        self.word_frequency = Counter()
        self.candidate_parents = set()
        
        logger.info(f"Processeur initialisé pour {tsv_file_path}")
        logger.info(f"  - Base URI pour la génération: {uri_base}")
        logger.info(f"  - Fréquence minimale pour candidats parents : {min_frequency_for_candidate}")

    def preprocess_data(self) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Exécute le pipeline complet de prétraitement des données.
        
        Le processus suit les étapes suivantes :
        
        1. Chargement des données TSV
        2. Nettoyage des définitions et extraction des sources
        3. Normalisation des labels
        4. Assignation d'URIs uniques (UUID)
        5. Création des index lexicaux
        6. Extraction des relations skos:broader existantes
        7. Analyse des relations depuis broader_from_preflabel
        8. Explosion des altLabel
        9. Calcul des statistiques
        
        Returns:
            tuple: Un triplet contenant :
            
            * **df** (*pd.DataFrame*) -- DataFrame nettoyé et préparé avec toutes les colonnes enrichies
            * **lexical_indexes** (*Dict[str, Any]*) -- Dictionnaire d'index lexicaux :
              
              - 'preflabel_to_uri': mapping prefLabel → URI
              - 'uri_to_preflabel': mapping URI → prefLabel  
              - 'altlabel_to_uri': mapping altLabel → list[URI]
              
            * **all_broader_relations** (*List[Dict[str, Any]]*) -- Liste consolidée de toutes les relations broader détectées
        
        Raises:
            FileNotFoundError: Si le fichier TSV source n'existe pas
            ValueError: Si les colonnes requises sont manquantes
            Exception: En cas d'erreur lors du traitement
        
        Note:
            Cette méthode modifie l'état interne de l'instance en peuplant les attributs
            `processed_data`, `lexical_indexes`, et les listes de relations.
            
        See Also:
            :meth:`get_statistics` : Pour obtenir des statistiques détaillées
            :meth:`save_processed_data` : Pour sauvegarder les résultats
        """

        logger.info("Début du prétraitement des données...")
        
        # Étape 1 : Chargement et préparation initiale
        df = self._load_data()
        
        # Étape 2 : Nettoyage des définitions et extraction des sources
        df = self.clean_dataframe(df)
        
        # Étape 3 : Normalisation des labels (avant l'assignation des URIs pour s'assurer que prefLabel_clean est prêt)
        df = self.normalize_labels(df) 
        
        # Étape 4 : Assignation des URIs uniques (UUID)
        logger.info("Étape 4 : Assignation des URIs uniques (UUID) aux termes")
        df = self._assign_unique_uris(df)
        
        # Étape 5 : Création des index lexicaux (maintenant avec des URIs garantis)
        lexical_indexes = self.create_lexical_indexes(df)
        self.lexical_indexes = lexical_indexes # Stocker pour find_uri_by_preflabel
        
        # Étape 6 : Extraction des relations skos:broader existantes
        logger.info("Étape 6 : Extraction des relations skos:broader existantes")
        self.existing_broader_relations = self._extract_existing_broader_relations(df)
        
        # Étape 6bis : Analyse des relations depuis broader_from_preflabel
        logger.info("Étape 6bis : Analyse des relations depuis broader_from_preflabel")
        self.preflabel_broader_relations = self._analyze_preflabel_broader_relations(df)
        
        # Consolidation de toutes les relations
        all_broader_relations = self.existing_broader_relations + self.preflabel_broader_relations
        logger.info(f"Relations consolidées: {len(self.existing_broader_relations)} existantes + "
                    f"{len(self.preflabel_broader_relations)} depuis prefLabel = "
                    f"{len(all_broader_relations)} total")
        
        # Étape 7 : Explosion des altLabel (après avoir extrait les relations existantes basées sur prefLabel)
        # Note: Cette étape duplique les lignes pour chaque altLabel. 
        # L'OutputGenerator devra gérer le regroupement pour la sortie finale "1 ligne par prefLabel".
        df = self.explode_altlabels(df)
        
        self.processed_data = df
        logger.info("Preprocessing terminé avec succès")

        # Étape 8 : Calcul des statistiques
        stats = self.get_statistics()
        
        # Affichage des statistiques
        logger.info("Statistiques finales :")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return df, lexical_indexes, all_broader_relations

    def _assign_unique_uris(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigne un URI unique (UUID) à chaque terme qui n'en a pas déjà un.
        Les URIs existants sont conservés s'ils ne sont pas NaN ou vides.
        
        Args:
            df: DataFrame avec les données brutes (doit contenir 'prefLabel' et 'URI')
            
        Returns:
            DataFrame avec une colonne 'URI' remplie pour chaque terme.
            
        Note:
            Les UUID v4 sont statistiquement uniques, aucune vérification de collision n'est nécessaire.
        """
        if 'URI' not in df.columns:
            df['URI'] = ''  # Assurer que la colonne existe
            
        # Compter les URIs à générer pour le log
        missing_uris = df['URI'].isna() | (df['URI'] == '')
        uris_to_generate = missing_uris.sum()
        
        if uris_to_generate > 0:
            logger.info(f"Génération de {uris_to_generate} nouveaux URIs")
        
        # Génération directe des URIs sans vérification de collision
        for idx, row in df.iterrows():
            current_uri = row['URI']
            
            # Si l'URI est manquant ou vide, générer un nouvel UUID
            if pd.isna(current_uri) or current_uri == '':
                df.loc[idx, 'URI'] = f"{self.uri_base}{uuid.uuid4()}"
        
        return df

    def _load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier TSV.
        
        Returns:
            DataFrame contenant les données brutes
            
        Raises:
            Exception: Si le fichier ne peut pas être lu ou ne contient pas les colonnes requises
        """
        try:
            df = pd.read_csv(
                self.tsv_file_path, 
                sep='\t', 
                encoding='utf-8',
                na_values=['', 'NaN', 'NULL'],  # Définir explicitement ce qui est NaN
                keep_default_na=False  # Ne pas utiliser les valeurs par défaut pandas
            )
            logger.info(f"Données chargées : {len(df)} lignes")
            
            # Vérification des colonnes requises
            required_columns = ['skos:prefLabel', 'skos:definition', 'skos:altLabel']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise Exception(f"Colonnes manquantes : {missing_columns}")
            
            self.raw_data = df
            return df.copy()
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données : {e}")
            raise

    def extract_sources(self, definition: str) -> Tuple[str, str]:
        """
        Extrait la source d'une définition (texte entre parenthèses finales).
        
        Cette méthode gère intelligemment les parenthèses imbriquées et ne considère
        comme source que le texte entre les dernières parenthèses fermées à la fin de la définition.
        
        Args:
            definition (str): Texte de la définition pouvant contenir une source entre parenthèses
            
        Returns:
            tuple: Un couple (*définition_nettoyée*, *source_extraite*)
            
            * Si une source est trouvée : (définition sans source, source)
            * Si pas de source : (définition originale, "")
        
        Examples:
            >>> processor = ThesaurusDataProcessor("test.tsv", "http://example.org/")
            >>> processor.extract_sources("Un concept important (Larousse 2023)")
            ('Un concept important', 'Larousse 2023')
            
            >>> processor.extract_sources("Concept (avec parenthèses) final (Source)")
            ('Concept (avec parenthèses) final', 'Source')
            
            >>> processor.extract_sources("Pas de source ici")
            ('Pas de source ici', '')
        
        Note:
            Les parenthèses imbriquées sont correctement gérées grâce à un comptage
            des parenthèses ouvrantes/fermantes.
        """
        if pd.isna(definition) or not isinstance(definition, str):
            return "", ""
        
        definition = definition.strip()
        
        # Chercher la dernière parenthèse ouvrante
        last_open_paren = definition.rfind('(')
        
        if last_open_paren == -1:
            # Pas de parenthèse ouvrante trouvée
            return definition, ""
        
        # Chercher la parenthèse fermante correspondante en partant de la fin
        last_close_paren = definition.rfind(')')
        
        if last_close_paren == -1 or last_close_paren <= last_open_paren:
            # Pas de parenthèse fermante ou mal placée
            return definition, ""
        
        # Vérifier que la parenthèse fermante est bien à la fin (éventuellement suivie d'espaces)
        after_close_paren = definition[last_close_paren + 1:].strip()
        if after_close_paren:
            # Il y a du texte après la parenthèse fermante, ce n'est pas une source finale
            return definition, ""
        
        # Compter les parenthèses pour s'assurer qu'on a la bonne paire
        open_count = 0
        actual_start = last_open_paren
        
        # Remonter pour trouver la vraie parenthèse ouvrante qui correspond à la fermante finale
        for i in range(last_close_paren - 1, -1, -1):
            if definition[i] == ')':
                open_count += 1
            elif definition[i] == '(':
                if open_count == 0:
                    actual_start = i
                    break
                open_count -= 1
        
        # Extraire la source et nettoyer la définition
        source = definition[actual_start + 1:last_close_paren].strip()
        clean_definition = definition[:actual_start].strip()
        
        return clean_definition, source
    
    def extract_broader_from_preflabel(self, preflabel: str) -> Tuple[str, Optional[str]]:
        """
        Extrait les informations skos:broader des prefLabel entre parenthèses finales.
        
        Args:
            preflabel: Label principal pouvant contenir des informations de hiérarchie
            
        Returns:
            Tuple (preflabel_nettoyé, broader_term)
            Si pas de broader, retourne (preflabel, None)
        """
        if pd.isna(preflabel) or not isinstance(preflabel, str):
            return "", None
        
        # Pattern similaire à extract_sources mais pour les prefLabel
        pattern = r'^(.*?)\s*\(([^)]+)\)\s*$'
        match = re.search(pattern, preflabel.strip())
        
        if match:
            clean_preflabel = match.group(1).strip()
            broader_term = match.group(2).strip()
            return clean_preflabel, broader_term
        else:
            return preflabel.strip(), None

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les dataframe en extrayant les sources et les broader depuis les prefLabel.
        
        Args:
            df: DataFrame contenant les données
            
        Returns:
            DataFrame avec colonnes 'definition_clean', 'source', 'prefLabel_clean' et 'broader_from_preflabel' ajoutées
        """
        df = df.copy()
        
        # Extraction des sources des définitions
        definitions_sources = df['skos:definition'].apply(self.extract_sources)
        df['definition_clean'] = definitions_sources.apply(lambda x: x[0])
        df['source'] = definitions_sources.apply(lambda x: x[1])

        preflabel_broader = df['skos:prefLabel'].apply(self.extract_broader_from_preflabel)
        df['preflabel_clean'] = preflabel_broader.apply(lambda x: x[0])
        df['broader_from_preflabel'] = preflabel_broader.apply(lambda x: x[1])
        
        logger.info(f"Définitions nettoyées : {len(df[df['definition_clean'] != ''])} avec définition")
        logger.info(f"Sources extraites : {len(df[df['source'] != ''])} avec source")
        
        return df

    def normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les labels prefLabel et altLabel.
        Crée une colonne 'preflabel_normalized'.
        """
        logger.info("Normalisation des labels...")
        df['preflabel_normalized'] = df['preflabel_clean'].apply(normalize_text)
        
        logger.info("Normalisation des labels terminée.")
        return df

    def explode_altlabels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Explose les altLabel multiples en lignes séparées.
        """
        logger.info("Explosion des altLabel pour les index (le DataFrame principal n'est pas modifié).")
        
        # Ne retourne pas le df_exploded ici, le df principal n'est pas affecté.
        # Les altLabel sont utilisés dans create_lexical_indexes.
        return df # Retourne le df non-explosé. L'explosion est faite pour les index seulement.


    def create_lexical_indexes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Crée des index lexicaux basés sur les prefLabel nettoyés et les URIs,
        en incluant aussi les altLabel.
        Cette méthode s'appuie désormais sur la colonne 'URI' qui est garantie unique pour chaque ligne.
        """
        lexical_indexes = {}
        
        preflabel_to_uri = {} # Un prefLabel_clean doit mapper à un seul URI ici si l'URI est unique par ligne
        uri_to_preflabel = {}
        altlabel_to_uri = defaultdict(list) # Un altLabel_clean peut mapper à plusieurs URIs
        
        # Assurez-vous que prefLabel_clean et URI sont disponibles et correctement remplis
        for _, row in df.iterrows():
            uri = row['URI'] # URI est garanti d'être présent et unique
            pref_clean = normalize_text(row['preflabel_clean'])
            
            if pd.notna(pref_clean) and pref_clean not in preflabel_to_uri: # S'assurer 1:1 pour prefLabel
                preflabel_to_uri[pref_clean] = uri
            
            uri_to_preflabel[uri] = pref_clean # Ceci doit être 1:1 après l'assignation des URIs
            
            # Traitement des altLabels
            if pd.notna(row['skos:altLabel']) and row['skos:altLabel'] != '':
                alt_labels_list = [normalize_text(label.strip()) for label in re.split(r'##|, ', row['skos:altLabel'])]
                for alt_label in alt_labels_list:
                    altlabel_to_uri[alt_label].append(uri) # Un altLabel peut pointer vers un URI

        # Simplifier altlabel_to_uri si un altLabel pointe toujours vers le même URI
        # Non, un altLabel peut être ambigu et pointer vers plusieurs URIs. Garder defaultdict(list)
        
        lexical_indexes['preflabel_to_uri'] = preflabel_to_uri
        lexical_indexes['uri_to_preflabel'] = uri_to_preflabel
        lexical_indexes['altlabel_to_uri'] = altlabel_to_uri 

        logger.info("Index lexicaux créés avec URIs (incluant altLabel).")
        return lexical_indexes


    def _extract_existing_broader_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extrait les relations skos:broader existantes du DataFrame original.
        Utilise les URIs pour identifier les termes.
        """
        relations = []
        if 'skos:broader' in df.columns:
            for _, row in df.iterrows():
                child_label = row['skos:prefLabel'] # Le prefLabel original
                child_uri = row['URI'] # L'URI est maintenant garanti unique
                
                if pd.notna(row['skos:broader']) and row['skos:broader'] != '':
                    parent_labels = re.split(r'##|, ', row['skos:broader'])
                    
                    for parent_label in parent_labels:
                        parent_label = parent_label.strip()

                        parent_label_clean, broader_from_label = self.extract_broader_from_preflabel(parent_label)
                        
                        # Tenter de trouver l'URI du parent par son prefLabel nettoyé
                        parent_uri = None
                        parent_uri = self.find_uri_by_preflabel(normalize_text(parent_label_clean))
                        
                        if child_uri and parent_uri:
                            relations.append({
                                'child': child_label,
                                'parent': parent_label,
                                'child_uri': child_uri,
                                'parent_uri': parent_uri,
                                'relation_category': 'existing_skos_broader',
                                'relation_detail': f'Relation skos:broader existante : {child_label} -> {parent_label}',
                                'confidence': 1.0,
                                'type': 'existing_parent',
                                'source': 'skos:broader_column'
                            })
                        else:
                            # Log un avertissement si un parent skos:broader existant n'a pas pu être résolu en URI
                            logger.warning(f"Relation skos:broader existante ignorée pour '{child_label}' -> '{parent_label}' (URI parent non trouvé).")
                            relations.append({
                                'child': child_label,
                                'parent': parent_label,
                                'child_uri': child_uri,
                                'parent_uri': parent_uri,
                                'relation_category': 'existing_skos_broader',
                                'relation_detail': f'Relation skos:broader existante : {child_label} -> {parent_label}',
                                'confidence': 1.0,
                                'type': 'candidate_parent',
                                'source': 'skos:broader_column'
                            })
        return relations
    
    def _analyze_preflabel_broader_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyse la colonne 'broader_from_preflabel' pour détecter les parents existants et candidats.
        Utilise la même logique que la phase 2 pour identifier les parents.
        
        Args:
            df: DataFrame contenant les données avec la colonne 'broader_from_preflabel'
            
        Returns:
            Liste des relations détectées depuis broader_from_preflabel
        """
        logger.info("Analyse des relations depuis broader_from_preflabel")
        
        relations = []
        
        # D'abord, analyser la fréquence des termes de broader_from_preflabel
        self._analyze_broader_word_frequency(df)
        
        # Index des termes existants pour recherche rapide
        preflabel_to_normalized = {
            row['preflabel_clean']: row['preflabel_normalized'] 
            for _, row in df.iterrows() 
            if pd.notna(row['preflabel_clean']) and pd.notna(row['preflabel_normalized'])
        }
        existing_normalized_terms = set(preflabel_to_normalized.values())
        
        existing_parent_matches = 0
        candidate_parent_matches = 0
        
        for _, row in df.iterrows():
            child_preflabel = row['preflabel_clean']
            child_uri = row['URI']
            broader_from_preflabel = row['broader_from_preflabel']
            
            if (not child_preflabel or not child_uri or 
                pd.isna(broader_from_preflabel) or broader_from_preflabel == ''):
                continue
            
            # Normaliser le terme parent extrait du prefLabel
            parent_normalized = normalize_text(broader_from_preflabel)
            
            if not parent_normalized:
                continue
            
            parent_preflabel = None
            parent_uri = None
            relation_type = None
            confidence = 1.0
            
            # 1. Vérifier si le parent existe comme terme autonome
            if parent_normalized in existing_normalized_terms:
                # Trouver le prefLabel correspondant et son URI
                for p_label, p_norm in preflabel_to_normalized.items():
                    if p_norm == parent_normalized:
                        parent_preflabel = p_label
                        parent_uri = self.lexical_indexes.get('preflabel_to_uri', {}).get(p_label)
                        relation_type = 'existing_parent'
                        existing_parent_matches += 1
                        confidence = 1.0
                        break
            
            # 2. Si pas trouvé ET que le parent a une fréquence suffisante, le proposer comme candidat
            elif (parent_normalized in self.word_frequency and 
                  self.word_frequency[parent_normalized] >= self.min_frequency_for_candidate):
                parent_preflabel = broader_from_preflabel  # Utiliser le terme original
                parent_uri = None  # L'URI sera générée en Phase 5 pour les nouveaux termes
                relation_type = 'candidate_parent'
                candidate_parent_matches += 1
                confidence = 0.8
                self.candidate_parents.add(parent_normalized)
            
            # Créer la relation si un parent a été trouvé
            if parent_preflabel:
                relation = {
                    'child': child_preflabel,
                    'child_uri': child_uri,
                    'parent': parent_preflabel,
                    'parent_uri': parent_uri,
                    'relation_category': 'preflabel_extraction',
                    'relation_detail': f'"{child_preflabel}" extrait de prefLabel -> parent "{parent_preflabel}"',
                    'confidence': confidence,
                    'type': relation_type,
                    'source': 'broader_from_preflabel',
                    'word_frequency': self.word_frequency.get(parent_normalized, 0)
                }
                relations.append(relation)
        
        logger.info("Analyse des relations depuis broader_from_preflabel terminée :")
        logger.info(f"  - {len(relations)} relations trouvées")
        logger.info(f"  - {existing_parent_matches} avec parents existants")
        logger.info(f"  - {candidate_parent_matches} avec parents candidats")
        logger.info(f"  - {len(self.candidate_parents)} parents candidats uniques détectés")
        
        if self.candidate_parents:
            logger.info(f"  - Exemples de parents candidats : {list(self.candidate_parents)[:10]}")
        
        return relations

    def _analyze_broader_word_frequency(self, df: pd.DataFrame) -> None:
        """
        Analyse la fréquence des termes dans broader_from_preflabel pour identifier les candidats parents.
        
        Args:
            df: DataFrame contenant les données du thésaurus
        """
        logger.info("Analyse de la fréquence des termes dans broader_from_preflabel")
        
        broader_counter = Counter()
        processed_terms = 0
        
        for _, row in df.iterrows():
            broader_term = row['broader_from_preflabel']
            
            if pd.notna(broader_term) and broader_term != '':
                normalized_broader = normalize_text(broader_term)
                if normalized_broader:
                    broader_counter[normalized_broader] += 1
                    processed_terms += 1
        
        # Mettre à jour le compteur global
        self.word_frequency.update(broader_counter)
        
        # Identifier les candidats parents potentiels
        frequent_broader_terms = {
            term: freq for term, freq in broader_counter.items() 
            if freq >= self.min_frequency_for_candidate
        }
        
        logger.info("Analyse de fréquence des termes broader_from_preflabel terminée :")
        logger.info(f"  - {processed_terms} termes broader traités")
        logger.info(f"  - {len(broader_counter)} termes broader uniques trouvés")
        logger.info(f"  - {len(frequent_broader_terms)} termes fréquents (>= {self.min_frequency_for_candidate})")
        
        if frequent_broader_terms:
            top_frequent = broader_counter.most_common(5)
            logger.info(f"  - Top 5 termes broader fréquents : {top_frequent}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la Phase 1 après traitement.
        Cette méthode doit être appelée après preprocess_data().
        
        Returns:
            Dict contenant les statistiques de la phase 1
        """
        if self.processed_data is None:
            logger.warning("Aucune donnée traitée disponible. Exécutez d'abord preprocess_data()")
            return {}
    
        # Calculer les statistiques basées sur les données traitées
        df = self.processed_data
        
        stats = {
            'total_terms': len(df),
            'terms_with_definitions': df['skos:definition'].notna().sum(),
            'terms_with_altlabels': df['skos:altLabel'].notna().sum(),
            'total_existing_broader_relations': len(self.existing_broader_relations),
            'existing_broader_with_resolved_uris': len([r for r in self.existing_broader_relations if r['parent_uri']]),
            'existing_broader_unresolved': len([r for r in self.existing_broader_relations if not r['parent_uri']]),
            'total_preflabel_broader_relations': len(self.preflabel_broader_relations),
            'preflabel_broader_existing_parents': len([r for r in self.preflabel_broader_relations if r['type'] == 'existing_parent']),
            'preflabel_broader_candidate_parents': len([r for r in self.preflabel_broader_relations if r['type'] == 'candidate_parent']),
            'total_candidate_parents_detected': len(self.candidate_parents),
            'terms_with_broader_from_preflabel': len([r for r in self.preflabel_broader_relations if r]),
            'uris_generated': len([row for _, row in df.iterrows() if row['URI'].startswith(self.uri_base)]),
            'average_word_frequency': sum(self.word_frequency.values()) / len(self.word_frequency) if self.word_frequency else 0,
            'top_candidate_parents': list(self.word_frequency.most_common(5)),
            'lexical_indexes_created': len(self.lexical_indexes),
            'preflabel_to_uri_mappings': len(self.lexical_indexes.get('preflabel_to_uri', {})),
            'altlabel_to_uri_mappings': len(self.lexical_indexes.get('altlabel_to_uri', {})),
        }
        
        return stats

    def save_processed_data(self, output_path: str, df: pd.DataFrame = None) -> None:
        """
        Sauvegarde les données préparées.
        """
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("Aucune donnée à sauvegarder. Exécutez d'abord preprocess_data()")
        
        df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        logger.info(f"Données sauvegardées dans : {output_path}")
    
    def get_existing_broader_relations(self) -> List[Dict[str, Any]]:
        """
        Retourne les relations skos:broader existantes extraites.
        """
        return self.existing_broader_relations
    
    def get_preflabel_broader_relations(self) -> List[Dict[str, Any]]:
        """
        Retourne les relations extraites depuis broader_from_preflabel.
        """
        return self.preflabel_broader_relations
    
    def get_all_broader_relations(self) -> List[Dict[str, Any]]:
        """
        Retourne toutes les relations broader (existantes + depuis prefLabel).
        """
        return self.existing_broader_relations + self.preflabel_broader_relations
    
    def get_candidate_parents(self) -> set:
        """
        Retourne les parents candidats détectés.
        """
        return self.candidate_parents
    
    def find_uri_by_preflabel(self, preflabel_normalized: str) -> Optional[str]:
        """
        Trouve l'URI d'un terme par son prefLabel_clean.
        """
        # Utilise l'index simplifié qui garantit 1:1 ou None si ambigu
        return self.lexical_indexes.get('preflabel_to_uri', {}).get(preflabel_normalized)
    
    def find_preflabel_by_uri(self, uri: str) -> Optional[str]:
        """
        Trouve le prefLabel_clean d'un terme par son URI.
        """
        return self.lexical_indexes.get('uri_to_preflabel', {}).get(uri)