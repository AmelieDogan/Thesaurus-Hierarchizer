"""
Module d'analyse sémantique par embeddings pour le thésaurus musical.

Ce module implémente la Phase 4 du cahier des charges avec :

- Embeddings spécialisés par domaine (français historique)
- Validation sémantique des relations détectées par les phases précédentes
- Détection des concepts abstraits liés
- Enrichissement des familles conceptuelles par clustering sémantique
- Intégration patterns-embeddings selon les règles spécifiées

Le module utilise des modèles de transformeurs pré-entraînés (comme CamemBERT) pour calculer
des embeddings sémantiques et effectuer des analyses de similarité cosinus entre termes.

Notes:
    - Les relations générées utilisent les champs normalisés ``relation_category`` et ``type``
    - Le champ ``type`` peut être 'existing_parent' ou 'candidate_parent'
    - Toutes les relations sont marquées avec ``validated_by_embedding`` (booléen)
    
Exemple:
    >>> analyzer = SemanticEmbeddingAnalyzer(
    ...     embedding_model_path="dangvantuan/sentence-camembert-base",
    ...     abstract_similarity_threshold=0.75,
    ...     validation_similarity_threshold=0.65,
    ...     family_similarity_threshold=0.70,
    ...     clustering_n_clusters=None,
    ...     clustering_distance_threshold=0.3
    ... )
    >>> relations, clusters = analyzer.analyze_embeddings(df, all_prev_relations, lexical_indexes)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from .logger import get_logger

logger = get_logger(__name__)

# Suppression des warnings pour les modèles
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers non disponible, utilisation d'embeddings simulés")
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers non disponible, utilisation d'embeddings simulés")
    HAS_TRANSFORMERS = False


class SemanticEmbeddingAnalyzer:
    """
    Classe responsable de l'analyse sémantique des termes du thésaurus à l'aide d'embeddings.
    
    Cette classe gère :
    
    - Le chargement ou la simulation des modèles d'embeddings
    - La validation sémantique des relations détectées par les phases précédentes
    - Le calcul des similarités sémantiques entre termes
    - La création de clusters sémantiques pour l'enrichissement
    - L'identification de concepts abstraits et familles conceptuelles
    - La génération de nouvelles relations candidates basées sur la similarité sémantique
    
    Le processus d'analyse comprend :
    
    1. **Validation des relations existantes** : Toutes les relations des phases précédentes
       sont évaluées sémantiquement et marquées avec ``validated_by_embedding``
    2. **Clustering sémantique** : Regroupement des termes similaires pour identifier
       des familles conceptuelles
    3. **Enrichissement conceptuel** : Proposition de nouvelles relations basées sur
       la proximité sémantique
    4. **Détection d'abstractions** : Identification de concepts génériques manquants
    
    Attributes:
        model_path (str): Chemin vers le modèle SentenceTransformer
        abstract_similarity_threshold (float): Seuil pour identifier les concepts abstraits
        validation_similarity_threshold (float): Seuil pour valider les relations existantes
        family_similarity_threshold (float): Seuil pour l'enrichissement de familles
        clustering_n_clusters (int): Nombre de clusters (optionnel)
        clustering_distance_threshold (float): Seuil de distance pour le clustering
        model: Instance du modèle SentenceTransformer chargé
        embeddings_cache (Dict[str, np.ndarray]): Cache des embeddings calculés
        semantic_clusters (List[Dict]): Clusters sémantiques générés
        stats (defaultdict): Statistiques de l'analyse
        preflabel_to_uri_map (Dict[str, str]): Mapping des labels vers URIs
        uri_to_preflabel_map (Dict[str, str]): Mapping des URIs vers labels
    """
    
    def __init__(self,
                 embedding_model_path: str,
                 abstract_similarity_threshold: float,
                 validation_similarity_threshold: float,
                 family_similarity_threshold: float,
                 clustering_n_clusters: int,
                 clustering_distance_threshold: float):
        """
        Initialise l'analyseur d'embeddings sémantiques.
        
        Args:
            embedding_model_path (str): Nom du modèle SentenceTransformer à utiliser 
                (ex: "dangvantuan/sentence-camembert-base")
            abstract_similarity_threshold (float): Seuil de similarité pour l'identification 
                de concepts abstraits (recommandé: 0.75-0.85)
            validation_similarity_threshold (float): Seuil de similarité pour valider 
                les relations par patterns (recommandé: 0.65-0.75)
            family_similarity_threshold (float): Seuil de similarité pour l'enrichissement 
                de familles conceptuelles (recommandé: 0.70-0.80)
            clustering_n_clusters (int): Nombre de clusters pour l'agglomération 
                (si None, utilise distance_threshold)
            clustering_distance_threshold (float): Seuil de distance pour l'agglomération 
                hiérarchique (recommandé: 0.2-0.4)
                
        Note:
            Si les librairies ``sentence-transformers`` ou ``transformers`` ne sont pas
            disponibles, des embeddings aléatoires seront utilisés pour les tests.
        """
        self.model_path = embedding_model_path
        self.abstract_similarity_threshold = abstract_similarity_threshold
        self.validation_similarity_threshold = validation_similarity_threshold
        self.family_similarity_threshold = family_similarity_threshold
        self.clustering_n_clusters = clustering_n_clusters
        self.clustering_distance_threshold = clustering_distance_threshold
        
        self.model = None
        self.embeddings_cache = {}  # Cache pour stocker les embeddings générés
        self.semantic_clusters = []
        self.stats = defaultdict(int)

        # Maps pour les URIs, définies lors de l'appel à analyze_embeddings
        self.preflabel_to_uri_map = {}
        self.uri_to_preflabel_map = {}
        
        self._load_model()
        
        logger.info(f"Analyseur sémantique initialisé avec modèle : {self.model_path}")
        logger.info(f"  - Seuil d'abstraction : {abstract_similarity_threshold}")
        logger.info(f"  - Seuil de validation : {validation_similarity_threshold}")
        
    def _load_model(self):
        """
        Charge le modèle SentenceTransformer ou configure un simulateur.
        
        Tente de charger le modèle spécifié. En cas d'échec ou si les dépendances
        ne sont pas disponibles, utilise des embeddings simulés (vecteurs aléatoires).
        
        Raises:
            Warning: Si le modèle ne peut pas être chargé ou si les dépendances manquent
        """
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.model_path)
                logger.info(f"Modèle SentenceTransformer '{self.model_path}' chargé avec succès.")
            except Exception as e:
                logger.error(f"Erreur de chargement du modèle SentenceTransformer '{self.model_path}': {e}. Utilisation d'embeddings simulés.")
                self.model = None
        else:
            logger.warning("sentence-transformers n'est pas installé. Les embeddings seront simulés.")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Génère ou récupère l'embedding d'un texte.
        
        Utilise un cache pour les embeddings déjà calculés afin d'optimiser les performances.
        Si le modèle n'est pas disponible, génère un vecteur aléatoire pour les tests.
        
        Args:
            text (str): Texte à encoder en embedding vectoriel
            
        Returns:
            np.ndarray: Vecteur d'embedding normalisé (dimension 768 par défaut)
            
        Note:
            Les embeddings sont mis en cache automatiquement pour éviter les recalculs.
        """
        if text in self.embeddings_cache:
            self.stats['embeddings_from_cache'] += 1
            return self.embeddings_cache[text]
        
        self.stats['embeddings_calculated'] += 1
        if self.model:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Erreur lors de l'encodage du texte '{text}': {e}. Génération d'un embedding aléatoire.")
                embedding = np.random.rand(768)  # Taille typique pour CamemBERT
        else:
            # Simulation d'embedding si le modèle n'est pas disponible
            embedding = np.random.rand(768)  # Vecteur aléatoire de dimension 768
            
        self.embeddings_cache[text] = embedding
        return embedding
    
    def calculate_similarity(self, term1: str, term2: str) -> float:
        """
        Calcule la similarité cosinus entre deux termes.
        
        Args:
            term1 (str): Premier terme à comparer
            term2 (str): Deuxième terme à comparer
            
        Returns:
            float: Score de similarité cosinus normalisé entre 0.0 et 1.0
                  (0.0 = très différent, 1.0 = très similaire)
                  
        Note:
            La similarité cosinus brute (entre -1 et 1) est normalisée vers [0, 1]
            en appliquant la transformation (similarity + 1) / 2.
        """
        if not term1 or not term2:
            return 0.0
            
        emb1 = self._get_embedding(term1)
        emb2 = self._get_embedding(term2)
        
        # Reshape for cosine_similarity function
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        
        # Normaliser la similarité de -1 à 1 vers 0 à 1
        return (similarity + 1) / 2
    
    def create_semantic_clusters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Crée des clusters sémantiques de termes en utilisant l'agglomération hiérarchique.
        
        Utilise la distance cosinus et un clustering hiérarchique pour regrouper
        les termes sémantiquement proches. Chaque cluster identifie une famille
        conceptuelle potentielle.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les termes avec colonnes 
                'preflabel_clean' et 'URI'
            
        Returns:
            List[Dict[str, Any]]: Liste de dictionnaires représentant les clusters.
                Chaque dictionnaire contient :
                
                - cluster_id (int): Identifiant unique du cluster
                - terms (List[str]): Liste des termes du cluster
                - uris (List[str]): Liste des URIs correspondantes
                - size (int): Nombre de termes dans le cluster
                - average_internal_similarity (float): Similarité moyenne interne
                
        Note:
            Seuls les clusters contenant plus d'un terme sont retournés.
            La similarité interne moyenne aide à évaluer la cohésion du cluster.
        """
        logger.info("Création des clusters sémantiques...")
        
        terms = df['preflabel_clean'].dropna().tolist()
        term_uris = df.set_index('preflabel_clean')['URI'].to_dict()  # Map prefLabel_clean to URI
        
        if not terms:
            logger.warning("Aucun terme disponible pour le clustering sémantique.")
            return []
            
        # Obtenir les embeddings pour tous les termes
        embeddings = np.array([self._get_embedding(term) for term in terms])
        
        if embeddings.shape[0] < 2:
            logger.warning("Moins de 2 termes pour le clustering sémantique.")
            return []

        # Clustering hiérarchique agglomératif
        # Utiliser linkage='average' pour la distance moyenne entre clusters
        clustering_model = AgglomerativeClustering(
            n_clusters=self.clustering_n_clusters, 
            distance_threshold=self.clustering_distance_threshold,
            metric='cosine',  # Utiliser la similarité cosinus comme métrique
            linkage='average'
        )
        
        # Convertir les embeddings en distances pour le clustering
        # La distance cosinus est 1 - similarité cosinus
        # AgglomerativeClustering avec metric='cosine' calcule automatiquement 1 - cos_sim
        labels = clustering_model.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(terms[i])
            
        semantic_clusters_output = []
        for cluster_id, cluster_terms in clusters.items():
            if len(cluster_terms) > 1:  # Ne retenir que les clusters avec plus d'un terme
                cluster_uris = [term_uris.get(term, None) for term in cluster_terms]
                
                # Calculer la similarité interne moyenne du cluster
                cluster_embeddings = np.array([self._get_embedding(term) for term in cluster_terms])
                avg_internal_similarity = 0.0
                if len(cluster_embeddings) > 1:
                    pairwise_similarities = cosine_similarity(cluster_embeddings)
                    # Exclure la diagonale (similarité d'un terme avec lui-même)
                    avg_internal_similarity = np.mean(pairwise_similarities[np.triu_indices(len(cluster_embeddings), k=1)])
                
                semantic_clusters_output.append({
                    'cluster_id': cluster_id,
                    'terms': cluster_terms,
                    'uris': cluster_uris,  # Inclure les URIs
                    'size': len(cluster_terms),
                    'average_internal_similarity': (avg_internal_similarity + 1) / 2  # Normaliser à 0-1
                })
                self.stats['clusters_found'] += 1
                self.stats['terms_in_clusters'] += len(cluster_terms)
        
        self.semantic_clusters = sorted(semantic_clusters_output, key=lambda x: x['size'], reverse=True)
        logger.info(f"Terminé. {len(self.semantic_clusters)} clusters sémantiques créés.")
        return self.semantic_clusters
    
    def validate_all_relations_with_embeddings(self, 
                                               all_prev_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Valide toutes les relations des phases précédentes à l'aide des embeddings sémantiques.
        
        Ajoute le champ ``validated_by_embedding`` (booléen) à chaque relation existante
        en calculant la similarité sémantique entre les termes enfant et parent.
        Les relations avec une similarité >= validation_similarity_threshold sont
        marquées comme validées (True).
        
        Args:
            all_prev_relations (List[Dict[str, Any]]): Liste complète des relations 
                détectées par toutes les phases précédentes (préexistantes, patterns, lexicales)
            
        Returns:
            List[Dict[str, Any]]: Liste des relations avec le nouveau champ 
                ``validated_by_embedding`` ajouté. Les autres champs sont préservés.
                
        Note:
            Cette méthode modifie les relations en place et ne crée pas de nouvelles relations.
            La validation sémantique complète (ajustement de confidence) sera effectuée
            dans une phase d'optimisation ultérieure.
        """
        logger.info(f"Validation sémantique de {len(all_prev_relations)} relations des phases précédentes...")
        validated_relations = []
        
        for relation in tqdm(all_prev_relations, desc="Validation sémantique", unit="relation"):
            child_term = relation.get('child')
            parent_term = relation.get('parent')
            
            # Si l'un des termes est manquant, marquer comme non validé
            if not child_term or not parent_term:
                relation['validated_by_embedding'] = False
                relation['semantic_similarity'] = 0.0
                self.stats['relations_skipped_missing_term'] += 1
                validated_relations.append(relation)
                continue

            # Calculer la similarité sémantique
            semantic_similarity = self.calculate_similarity(child_term, parent_term)
            
            # Déterminer la validation basée sur le seuil
            is_semantically_validated = semantic_similarity >= self.validation_similarity_threshold
            
            # Ajouter les nouveaux champs à la relation existante
            relation['validated_by_embedding'] = is_semantically_validated
            relation['semantic_similarity'] = semantic_similarity
            
            # Mise à jour des statistiques
            if is_semantically_validated:
                self.stats['semantic_validation_success'] += 1
            else:
                self.stats['semantic_validation_failure'] += 1
            
            validated_relations.append(relation)
        
        logger.info(f"Terminé. {self.stats['semantic_validation_success']} relations validées sémantiquement, "
                   f"{self.stats['semantic_validation_failure']} relations non validées.")
        return validated_relations
    
    def enrich_concept_families(self, 
                                df: pd.DataFrame, 
                                existing_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifie et propose des relations pour des familles conceptuelles basées sur le clustering.
        
        Utilise les clusters sémantiques pour identifier des termes apparentés
        et propose de nouvelles relations hiérarchiques. Le terme le plus court
        et sémantiquement représentatif de chaque cluster est proposé comme parent.
        
        Args:
            df (pd.DataFrame): DataFrame complet des termes avec URIs
            existing_relations (List[Dict[str, Any]]): Relations déjà détectées 
                pour éviter les doublons
            
        Returns:
            List[Dict[str, Any]]: Liste des nouvelles relations enrichies.
                Chaque relation contient :
                
                - child/child_uri: Terme et URI enfant
                - parent/parent_uri: Terme et URI parent (peut être None si candidat)
                - relation_category: 'semantic_cluster_shortest_term'
                - type: 'existing_parent' ou 'candidate_parent'
                - confidence: Similarité moyenne du cluster
                - validated_by_embedding: True (par défaut pour les relations générées)
                - source: 'embedding'
                - cluster_id: Identifiant du cluster source
                
        Note:
            Seuls les clusters avec une similarité interne >= family_similarity_threshold
            et contenant au moins 2 termes sont considérés pour l'enrichissement.
        """
        logger.info("Enrichissement des familles conceptuelles par analyse sémantique...")
        new_semantic_relations = []
        
        # Utiliser les clusters sémantiques déjà calculés
        if not self.semantic_clusters:
            self.create_semantic_clusters(df)
            
        # Créer un set des relations existantes pour éviter les doublons
        existing_rel_tuples = set()
        for rel in existing_relations:
            if rel.get('child_uri') and rel.get('parent_uri'):
                existing_rel_tuples.add((rel['child_uri'], rel['parent_uri']))
            else:  # Fallback pour les relations sans URI complètes
                existing_rel_tuples.add((rel['child'], rel['parent']))

        # Pour chaque cluster sémantique
        for cluster in self.semantic_clusters:
            cluster_terms = cluster['terms']
            cluster_uris = cluster['uris']
            avg_sim = cluster['average_internal_similarity']
            
            if avg_sim >= self.family_similarity_threshold and len(cluster_terms) >= 2:
                # Identifier le terme le plus court comme parent potentiel
                shortest_term = sorted(cluster_terms, key=len)[0]
                shortest_term_uri = self.preflabel_to_uri_map.get(shortest_term)
                
                # Vérifier si le terme le plus court est sémantiquement représentatif
                is_representative = True
                for term in cluster_terms:
                    if term != shortest_term:
                        sim_to_shortest = self.calculate_similarity(term, shortest_term)
                        if sim_to_shortest < self.family_similarity_threshold:
                            is_representative = False
                            break
                            
                if is_representative:
                    # Déterminer le type de relation basé sur l'existence de l'URI parent
                    relation_type = 'existing_parent' if shortest_term_uri else 'candidate_parent'
                    
                    # Créer des relations parent-enfant pour tous les autres termes du cluster
                    for child_term, child_uri in zip(cluster_terms, cluster_uris):
                        if child_term != shortest_term:
                            relation_key = (child_uri, shortest_term_uri) if child_uri and shortest_term_uri else (child_term, shortest_term)
                            
                            if relation_key not in existing_rel_tuples:
                                new_semantic_relations.append({
                                    'child': child_term,
                                    'child_uri': child_uri,
                                    'parent': shortest_term,
                                    'parent_uri': shortest_term_uri,
                                    'relation_category': 'semantic_cluster_shortest_term',
                                    'type': relation_type,
                                    'confidence': avg_sim,
                                    'validated_by_embedding': True,  # True par défaut pour les relations générées
                                    'source': 'embedding',
                                    'semantic_similarity': avg_sim,
                                    'source_phase': 'semantic_enrichment',
                                    'cluster_id': cluster['cluster_id']
                                })
                                self.stats['semantic_relations_generated'] += 1
            else:
                self.stats['clusters_below_threshold'] += 1
        
        logger.info(f"Terminé. {len(new_semantic_relations)} nouvelles relations sémantiques générées.")
        return new_semantic_relations
    
    def identify_abstract_concepts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifie des concepts abstraits ou génériques potentiels non présents dans le thésaurus.
        
        Analyse les clusters sémantiques très cohésifs pour proposer des concepts
        abstraits manquants. Génère automatiquement des noms de concepts basés
        sur l'analyse des mots les plus fréquents dans chaque cluster.
        
        Args:
            df (pd.DataFrame): DataFrame complet des termes avec URIs
            
        Returns:
            List[Dict[str, Any]]: Liste des relations vers des concepts abstraits proposés.
                Chaque relation contient :
                
                - child/child_uri: Terme spécifique existant
                - parent/parent_uri: Concept abstrait proposé (URI sera None)
                - relation_category: 'semantic_abstract_concept'
                - type: 'candidate_parent' (le parent abstrait n'existe pas encore)
                - confidence: Similarité moyenne du cluster
                - validated_by_embedding: True
                - source: 'embedding'
                - cluster_id: Identifiant du cluster source
                
        Note:
            Seuls les clusters avec une similarité >= abstract_similarity_threshold
            et contenant au moins 3 termes sont considérés. Les concepts abstraits
            proposés sont basés sur les mots les plus fréquents dans le cluster.
        """
        logger.info("Identification des concepts abstraits potentiels...")
        abstract_concept_relations = []
        
        # Utiliser les clusters sémantiques déjà calculés
        if not self.semantic_clusters:
            self.create_semantic_clusters(df)

        for cluster in self.semantic_clusters:
            cluster_terms = cluster['terms']
            cluster_uris = cluster['uris']
            avg_sim = cluster['average_internal_similarity']
            
            # Si le cluster est très cohésif mais n'a pas de parent clair existant
            if avg_sim >= self.abstract_similarity_threshold and len(cluster_terms) >= 3:
                # Générer un nom pour le concept abstrait basé sur les mots fréquents
                word_counter = Counter()
                for term in cluster_terms:
                    word_counter.update(term.lower().split())
                
                most_common_word = None
                if word_counter:
                    # Prendre le mot le plus fréquent avec au moins 3 caractères
                    for word, count in word_counter.most_common():
                        if len(word) >= 3:
                            most_common_word = word
                            break
                
                if most_common_word:
                    # Vérifier que ce concept abstrait n'existe pas déjà dans le thésaurus
                    if most_common_word.lower() not in df['preflabel_clean'].str.lower().values:
                        abstract_parent_label = most_common_word.capitalize()
                        
                        # Créer des relations pour chaque terme du cluster vers ce parent abstrait
                        for child_term, child_uri in zip(cluster_terms, cluster_uris):
                            abstract_concept_relations.append({
                                'child': child_term,
                                'child_uri': child_uri,
                                'parent': abstract_parent_label,
                                'parent_uri': None,  # URI sera générée en Phase 5
                                'relation_category': 'semantic_abstract_concept',
                                'type': 'candidate_parent',  # Le parent abstrait n'existe pas encore
                                'confidence': avg_sim,
                                'validated_by_embedding': True,  # True par défaut
                                'source': 'embedding',
                                'semantic_similarity': avg_sim,
                                'source_phase': 'semantic_enrichment',
                                'cluster_id': cluster['cluster_id']
                            })
                            self.stats['abstract_concepts_proposed'] += 1
                            self.stats['abstract_concept_relations_generated'] += 1
        
        logger.info(f"Terminé. {len(abstract_concept_relations)} relations de concepts abstraits proposées.")
        return abstract_concept_relations

    def analyze_embeddings(self, 
                           df: pd.DataFrame, 
                           all_prev_relations: List[Dict[str, Any]],
                           lexical_indexes: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Exécute l'analyse complète des embeddings sémantiques.
        
        Cette méthode constitue le point d'entrée principal de la phase 4. Elle effectue
        l'ensemble du processus d'analyse sémantique en plusieurs étapes :
        
        1. **Validation des relations existantes** : Toutes les relations des phases 
           précédentes sont évaluées sémantiquement
        2. **Clustering sémantique** : Regroupement des termes par similarité
        3. **Enrichissement conceptuel** : Génération de nouvelles relations basées 
           sur les clusters
        4. **Détection d'abstractions** : Identification de concepts génériques manquants
        
        Args:
            df (pd.DataFrame): DataFrame du thésaurus avec colonnes 'URI', 'preflabel_clean', etc.
            all_prev_relations (List[Dict[str, Any]]): Toutes les relations détectées par 
                les phases précédentes (préexistantes, patterns lexicaux, familles lexicales)
            lexical_indexes (Dict[str, Any]): Dictionnaire des index lexicaux incluant 
                'uri_to_preflabel' et 'preflabel_to_uri'
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Tuple contenant :
            
            - **relations_semantiques** : Liste complète des relations avec validation 
              sémantique et nouvelles relations générées
            - **clusters_semantiques** : Liste des clusters sémantiques créés avec 
              leurs statistiques
              
        Note:
            Toutes les relations retournées incluent le champ ``validated_by_embedding`` (booléen)
            et utilisent les champs normalisés ``relation_category`` et ``type``.
        """
        logger.info("Début de l'analyse sémantique par embeddings.")

        # Initialiser les mappings URI <-> preflabel
        self.preflabel_to_uri_map = lexical_indexes.get('preflabel_to_uri', {})
        self.uri_to_preflabel_map = lexical_indexes.get('uri_to_preflabel', {})
        
        # Pré-calcul des embeddings pour optimiser les performances
        all_preflabels_clean = df['preflabel_clean'].dropna().tolist()
        logger.info(f"Pré-calcul des embeddings pour {len(all_preflabels_clean)} termes...")
        for term in tqdm(all_preflabels_clean, desc="Pré-calcul des embeddings", unit="terme"):
            self._get_embedding(term)
        logger.info("Pré-calcul des embeddings terminé.")

        # 1. Création des clusters sémantiques
        semantic_clusters = self.create_semantic_clusters(df)
        
        # 2. Validation de toutes les relations précédentes avec les embeddings
        validated_relations = self.validate_all_relations_with_embeddings(all_prev_relations)
        
        # 3. Enrichissement des familles conceptuelles basé sur les clusters
        semantic_enrichment_relations = self.enrich_concept_families(df, validated_relations)
        
        # 4. Identification de concepts abstraits manquants
        abstract_concept_relations = self.identify_abstract_concepts(df)
        
        # 5. Consolidation de toutes les relations sémantiques
        all_semantic_relations = validated_relations + semantic_enrichment_relations + abstract_concept_relations
        
        self.stats['total_semantic_relations'] = len(all_semantic_relations)
        self.stats['validated_relations'] = len(validated_relations)
        self.stats['new_enrichment_relations'] = len(semantic_enrichment_relations)
        self.stats['new_abstract_relations'] = len(abstract_concept_relations)
        
        logger.info(f"Analyse sémantique terminée. Total relations sémantiques: {len(all_semantic_relations)}")
        logger.info(f"  - Relations validées des phases précédentes: {len(validated_relations)}")
        logger.info(f"  - Nouvelles relations d'enrichissement: {len(semantic_enrichment_relations)}")
        logger.info(f"  - Nouvelles relations de concepts abstraits: {len(abstract_concept_relations)}")
        
        return all_semantic_relations, semantic_clusters

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées de l'analyse sémantique.
        
        Fournit un aperçu complet des performances et résultats de l'analyse,
        incluant les métriques de validation, d'enrichissement et de clustering.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant les statistiques suivantes :
            
            **Configuration** :
            
            - model_path (str): Chemin du modèle utilisé
            - abstract_similarity_threshold (float): Seuil pour concepts abstraits
            - validation_similarity_threshold (float): Seuil de validation
            - family_similarity_threshold (float): Seuil d'enrichissement
            
            **Performance des embeddings** :
            
            - embeddings_calculated (int): Nouveaux embeddings calculés
            - embeddings_from_cache (int): Embeddings récupérés du cache
            - total_embeddings_generated (int): Total des embeddings utilisés
            
            **Validation sémantique** :
            
            - semantic_validation_success (int): Relations validées positivement
            - semantic_validation_failure (int): Relations non validées
            - relations_skipped_missing_term (int): Relations ignorées (termes manquants)
            
            **Clustering sémantique** :
            
            - clusters_found (int): Nombre de clusters créés
            - terms_in_clusters (int): Nombre total de termes dans les clusters
            - clusters_below_threshold (int): Clusters non retenus
            
            **Génération de relations** :
            
            - semantic_relations_generated (int): Relations d'enrichissement créées
            - abstract_concepts_proposed (int): Concepts abstraits proposés
            - abstract_concept_relations_generated (int): Relations vers concepts abstraits
            
            **Totaux** :
            
            - total_semantic_relations (int): Nombre total de relations finales
            - validated_relations (int): Relations des phases précédentes validées
            - new_enrichment_relations (int): Nouvelles relations d'enrichissement
            - new_abstract_relations (int): Nouvelles relations de concepts abstraits
        """
        stats = dict(self.stats)  # Convertir defaultdict en dict
        stats.update({
            'model_path': self.model_path,
            'abstract_similarity_threshold': self.abstract_similarity_threshold,
            'validation_similarity_threshold': self.validation_similarity_threshold,
            'family_similarity_threshold': self.family_similarity_threshold,
            'total_embeddings_generated': stats.get('embeddings_calculated', 0) + stats.get('embeddings_from_cache', 0),
        })
        return stats
    
    def reset_cache(self) -> None:
        """
        Réinitialise le cache des embeddings.
        
        Vide le cache des embeddings précédemment calculés, forçant le recalcul
        lors des prochains appels. Utile pour libérer la mémoire ou forcer
        le recalcul avec un nouveau modèle.
        
        Note:
            Cette opération peut impacter les performances si de nombreux embeddings
            doivent être recalculés après la réinitialisation.
        """
        self.embeddings_cache = {}
        logger.info("Cache des embeddings réinitialisé")
    
    def get_cluster_details(self, cluster_id: int) -> Dict[str, Any]:
        """
        Retourne les détails d'un cluster sémantique spécifique.
        
        Args:
            cluster_id (int): Identifiant du cluster à analyser
            
        Returns:
            Dict[str, Any]: Détails du cluster ou None si non trouvé.
                Contient les mêmes champs que create_semantic_clusters()
                
        Example:
            >>> analyzer = SemanticEmbeddingAnalyzer(...)
            >>> cluster_details = analyzer.get_cluster_details(0)
            >>> print(f"Cluster contient {cluster_details['size']} termes")
        """
        for cluster in self.semantic_clusters:
            if cluster['cluster_id'] == cluster_id:
                return cluster
        return None
    
    def export_semantic_network(self) -> Dict[str, Any]:
        """
        Exporte une représentation du réseau sémantique pour visualisation.
        
        Crée une structure de données adaptée pour la visualisation de graphes
        des relations sémantiques découvertes.
        
        Returns:
            Dict[str, Any]: Structure contenant :
            
            - nodes (List[Dict]): Liste des nœuds (termes) avec leurs propriétés
            - edges (List[Dict]): Liste des arêtes (relations) avec leurs poids
            - clusters (List[Dict]): Information sur les clusters sémantiques
            - statistics (Dict): Statistiques du réseau
            
        Note:
            Cette méthode est utile pour créer des visualisations interactives
            du thésaurus enrichi avec des outils comme NetworkX, D3.js, ou Gephi.
        """
        nodes = []
        edges = []
        
        # Ajouter les nœuds basés sur les clusters
        node_id_map = {}
        node_counter = 0
        
        for cluster in self.semantic_clusters:
            for term, uri in zip(cluster['terms'], cluster['uris']):
                if term not in node_id_map:
                    nodes.append({
                        'id': node_counter,
                        'label': term,
                        'uri': uri,
                        'cluster_id': cluster['cluster_id'],
                        'cluster_similarity': cluster['average_internal_similarity']
                    })
                    node_id_map[term] = node_counter
                    node_counter += 1
        
        # Ajouter les arêtes basées sur les relations dans les clusters
        for cluster in self.semantic_clusters:
            cluster_terms = cluster['terms']
            # Créer des arêtes entre tous les termes du cluster
            for i, term1 in enumerate(cluster_terms):
                for j, term2 in enumerate(cluster_terms[i+1:], i+1):
                    if term1 in node_id_map and term2 in node_id_map:
                        similarity = self.calculate_similarity(term1, term2)
                        edges.append({
                            'source': node_id_map[term1],
                            'target': node_id_map[term2],
                            'weight': similarity,
                            'cluster_id': cluster['cluster_id']
                        })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'clusters': self.semantic_clusters,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_clusters': len(self.semantic_clusters),
                'average_cluster_size': np.mean([c['size'] for c in self.semantic_clusters]) if self.semantic_clusters else 0
            }
        }