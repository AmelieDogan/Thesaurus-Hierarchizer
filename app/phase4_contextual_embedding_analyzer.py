"""
Module de découverte sémantique contextuelle pour le thésaurus musical.

Ce module implémente la Phase 4 transformée du cahier des charges avec :

- Enrichissement contextuel des embeddings par relations existantes
- Pattern mining des relations pour découverte par analogie
- Graph-enhanced discovery par proximité structurelle
- Identification de zones sémantiques cohérentes
- Découverte de nouvelles relations hiérarchiques précises

Le module exploite les relations existantes comme contexte d'apprentissage
pour proposer de nouvelles relations inédites plutôt que de simplement
valider les relations précédentes.

Notes:
    - Les relations découvertes utilisent ``discovery_method`` et ``contextual_similarity``
    - Le champ ``type`` peut être 'existing_parent' ou 'candidate_parent'
    - Toutes les nouvelles relations incluent ``supporting_evidence``
    
Exemple:
    >>> analyzer = ContextualSemanticDiscoveryEngine(
    ...     embedding_model_path="dangvantuan/sentence-camembert-base",
    ...     context_window_size=50,
    ...     pattern_min_frequency=3,
    ...     analogy_similarity_threshold=0.70,
    ...     zone_coherence_threshold=0.75
    ... )
    >>> relations, zones, contexts = analyzer.analyze_embeddings(df, all_prev_relations, lexical_indexes)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import networkx as nx
import re

from .utils import normalize_text

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


class ContextualSemanticDiscoveryEngine:
    """
    Moteur de découverte sémantique contextuelle pour thésaurus musical.
    
    Cette classe transforme l'approche de validation simple en un système
    de découverte proactive qui exploite les relations existantes comme
    contexte d'apprentissage pour identifier de nouvelles relations précises.
    
    Le processus de découverte comprend :
    
    1. **Enrichissement contextuel** : Chaque terme est contextualisé par ses
       relations connues, sa définition et ses synonymes
    2. **Pattern mining relationnel** : Extraction de règles récurrentes des
       relations existantes pour découverte par analogie
    3. **Graph-enhanced discovery** : Construction d'un graphe de relations
       pour identifier des connexions par proximité structurelle
    4. **Zones sémantiques** : Identification de clusters conceptuels cohérents
       pour enrichissement intra-zones
    5. **Découverte multi-méthodes** : Combinaison des approches pour maximiser
       la précision des nouvelles relations proposées
    
    Attributes:
        model_path (str): Chemin vers le modèle d'embeddings
        context_window_size (int): Taille maximale du contexte enrichi (mots)
        pattern_min_frequency (int): Fréquence minimale pour retenir un pattern
        graph_walk_length (int): Longueur des marches aléatoires pour graph embeddings
        zone_coherence_threshold (float): Seuil de cohérence des zones sémantiques
        analogy_similarity_threshold (float): Seuil de similarité pour analogies
        model: Instance du modèle SentenceTransformer
        relation_graph (nx.DiGraph): Graphe dirigé des relations
        contextual_embeddings_cache (Dict[str, np.ndarray]): Cache des embeddings contextuels
        extracted_patterns (List[Dict]): Patterns relationnels extraits
        semantic_zones (List[Dict]): Zones sémantiques identifiées
        stats (defaultdict): Statistiques détaillées de la découverte
    """
    
    def __init__(self,
                 embedding_model_path: str,
                 context_window_size: int,
                 pattern_min_frequency: int,
                 graph_walk_length: int,
                 zone_coherence_threshold: float,
                 analogy_similarity_threshold: float,
                 abstract_similarity_threshold: float,
                 family_similarity_threshold: float):
        """
        Initialise le moteur de découverte sémantique contextuelle.
        
        Args:
            embedding_model_path (str): Nom du modèle SentenceTransformer à utiliser
            context_window_size (int): Nombre maximum de mots pour le contexte enrichi
            pattern_min_frequency (int): Fréquence minimale pour qu'un pattern soit retenu
            graph_walk_length (int): Longueur des marches aléatoires pour graph embeddings
            zone_coherence_threshold (float): Seuil de cohérence pour les zones sémantiques
            analogy_similarity_threshold (float): Seuil de similarité pour les analogies
            abstract_similarity_threshold (float): Seuil pour identifier concepts abstraits
            family_similarity_threshold (float): Seuil pour enrichissement de familles
        """
        self.model_path = embedding_model_path
        self.context_window_size = context_window_size
        self.pattern_min_frequency = pattern_min_frequency
        self.graph_walk_length = graph_walk_length
        self.zone_coherence_threshold = zone_coherence_threshold
        self.analogy_similarity_threshold = analogy_similarity_threshold
        self.abstract_similarity_threshold = abstract_similarity_threshold
        self.family_similarity_threshold = family_similarity_threshold
        
        self.model = None
        self.relation_graph = nx.DiGraph()
        self.contextual_embeddings_cache = {}
        self.extracted_patterns = []
        self.semantic_zones = []
        self.stats = defaultdict(int)
        
        # Maps pour les URIs
        self.preflabel_to_uri_map = {}
        self.uri_to_preflabel_map = {}
        self.altlabel_to_uri_map = {}
        
        self._load_model()
        
        logger.info(f"Moteur de découverte sémantique initialisé avec modèle : {self.model_path}")
        logger.info(f"  - Taille fenêtre contextuelle : {context_window_size}")
        logger.info(f"  - Seuil analogies : {analogy_similarity_threshold}")
        logger.info(f"  - Seuil zones sémantiques : {zone_coherence_threshold}")
        
    def _load_model(self):
        """
        Charge le modèle SentenceTransformer ou configure un simulateur.
        
        Raises:
            Warning: Si le modèle ne peut pas être chargé ou si les dépendances manquent
        """
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(self.model_path)
                logger.info(f"Modèle SentenceTransformer '{self.model_path}' chargé avec succès.")
            except Exception as e:
                logger.error(f"Erreur de chargement du modèle '{self.model_path}': {e}. Utilisation d'embeddings simulés.")
                self.model = None
        else:
            logger.warning("sentence-transformers n'est pas installé. Les embeddings seront simulés.")
    
    def build_term_context(self, 
                          term: str, 
                          df: pd.DataFrame, 
                          all_relations: List[Dict[str, Any]]) -> str:
        """
        Construit le contexte textuel enrichi d'un terme basé sur ses relations et définitions.
        
        Crée un contexte sémantique riche en combinant :
        - Le terme principal
        - Sa définition nettoyée
        - Ses parents connus dans les relations
        - Ses enfants connus
        - Ses synonymes (altLabels)
        - Les termes "frères" (ayant les mêmes parents)
        
        Args:
            term (str): Terme à contextualiser
            df (pd.DataFrame): DataFrame contenant les informations des termes
            all_relations (List[Dict[str, Any]]): Toutes les relations existantes
            
        Returns:
            str: Contexte textuel enrichi limité à context_window_size mots
            
        Example:
            >>> context = analyzer.build_term_context("Absoute", df, relations)
            >>> print(context)
            "Absoute liturgie des morts cérémonie religieuse funéraire service église..."
        """
        context_words = [term]  # Commencer par le terme lui-même
        
        # Récupérer les informations du terme dans le DataFrame
        term_row = df[df['preflabel_clean'] == term]
        
        if not term_row.empty:
            # Ajouter la définition si disponible
            definition = term_row.iloc[0].get('definition_clean', '')
            if definition and pd.notna(definition):
                context_words.extend(definition.split())
            
            # Ajouter les synonymes (altLabels)
            alt_labels = term_row.iloc[0].get('skos:altLabel', '')
            if alt_labels and pd.notna(alt_labels):
                # Nettoyer et séparer les altLabels
                alt_terms = re.split(r'[,;|]', str(alt_labels))
                for alt_term in alt_terms:
                    alt_term = alt_term.strip()
                    if alt_term:
                        context_words.extend(alt_term.split())
        
        # Analyser les relations pour enrichir le contexte
        parents = set()
        children = set()
        
        for relation in all_relations:
            if relation.get('child') == term:
                parent = relation.get('parent')
                if parent:
                    parents.add(parent)
                    context_words.extend(parent.split())
            elif relation.get('parent') == term:
                child = relation.get('child')
                if child:
                    children.add(child)
                    context_words.extend(child.split())
        
        # Ajouter les termes "frères" (ayant les mêmes parents)
        for relation in all_relations:
            if relation.get('parent') in parents and relation.get('child') != term:
                sibling = relation.get('child')
                if sibling:
                    context_words.extend(sibling.split())
        
        # Limiter à la taille de fenêtre spécifiée
        context_words = context_words[:self.context_window_size]
        
        # Nettoyer et rejoindre
        clean_context = ' '.join([word.strip() for word in context_words if word.strip()])
        
        self.stats['contexts_built'] += 1
        return clean_context
    
    def create_contextual_embedding(self, 
                                   term: str, 
                                   df: pd.DataFrame, 
                                   all_relations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Crée un embedding enrichi par le contexte relationnel et définitionnel.
        
        Génère un embedding vectoriel basé sur le contexte enrichi du terme
        plutôt que sur le terme seul, capturant ainsi sa position sémantique
        dans le réseau de relations existantes.
        
        Args:
            term (str): Terme à encoder avec son contexte
            df (pd.DataFrame): DataFrame contenant les informations des termes
            all_relations (List[Dict[str, Any]]): Relations existantes pour contexte
            
        Returns:
            np.ndarray: Vecteur d'embedding contextuel normalisé
            
        Note:
            Les embeddings contextuels sont mis en cache pour optimiser les performances.
        """
        if term in self.contextual_embeddings_cache:
            self.stats['contextual_embeddings_from_cache'] += 1
            return self.contextual_embeddings_cache[term]
        
        # Construire le contexte enrichi
        enriched_context = self.build_term_context(term, df, all_relations)
        
        # Générer l'embedding du contexte enrichi
        if self.model:
            try:
                embedding = self.model.encode(enriched_context, convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                logger.error(f"Erreur lors de l'encodage contextuel de '{term}': {e}. Génération d'un embedding aléatoire.")
                embedding = np.random.rand(768)
        else:
            # Simulation d'embedding si le modèle n'est pas disponible
            embedding = np.random.rand(768)
            
        self.contextual_embeddings_cache[term] = embedding
        self.stats['contextual_embeddings_calculated'] += 1
        
        return embedding
    
    def extract_relation_patterns(self, all_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrait les patterns récurrents des relations existantes pour découverte par analogie.
        
        Analyse les relations existantes pour identifier des règles récurrentes
        du type "Si terme A contient pattern X ET A→B, alors autres termes
        contenant X peuvent→B".
        
        Args:
            all_relations (List[Dict[str, Any]]): Toutes les relations existantes
            
        Returns:
            List[Dict[str, Any]]: Liste des patterns extraits avec leurs statistiques.
                Chaque pattern contient :
                
                - pattern_type (str): Type de pattern (ex: 'suffix', 'prefix', 'contains')
                - pattern_text (str): Texte du pattern
                - target_parents (List[str]): Parents cibles associés
                - frequency (int): Fréquence d'occurrence
                - confidence (float): Confiance statistique
                - examples (List[Dict]): Exemples de relations supportant le pattern
        """
        logger.info("Extraction des patterns relationnels...")
        
        pattern_stats = defaultdict(lambda: defaultdict(list))
        
        # Analyser chaque relation pour extraire des patterns
        for relation in all_relations:
            child = relation.get('child', '').strip()
            parent = relation.get('parent', '').strip()
            
            if not child or not parent:
                continue
                
            # Patterns de suffixe (ex: "X-ballet" → "Ballet")
            if '-' in child:
                parts = child.split('-')
                if len(parts) == 2:
                    suffix_pattern = f"*-{parts[1]}"
                    if parts[1].lower() in parent.lower():
                        pattern_stats['suffix'][suffix_pattern].append((child, parent, relation))
            
            # Patterns de préfixe (ex: "Anche X" → "Anche")
            child_words = child.split()
            parent_words = parent.split()
            
            if len(child_words) > 1 and len(parent_words) >= 1:
                first_child_word = child_words[0]
                if first_child_word.lower() in [pw.lower() for pw in parent_words]:
                    prefix_pattern = f"{first_child_word} *"
                    pattern_stats['prefix'][prefix_pattern].append((child, parent, relation))
            
            # Patterns de contenu (mots clés récurrents)
            for word in child.split():
                if len(word) >= 4:  # Ignorer les mots très courts
                    if word.lower() in parent.lower():
                        contains_pattern = f"*{word}*"
                        pattern_stats['contains'][contains_pattern].append((child, parent, relation))
        
        # Filtrer et formater les patterns selon la fréquence minimale
        extracted_patterns = []
        
        for pattern_type, patterns in pattern_stats.items():
            for pattern_text, occurrences in patterns.items():
                frequency = len(occurrences)
                
                if frequency >= self.pattern_min_frequency:
                    # Calculer la confiance et les parents cibles
                    target_parents = [occ[1] for occ in occurrences]
                    parent_counts = Counter(target_parents)
                    most_common_parent = parent_counts.most_common(1)[0][0]
                    confidence = parent_counts[most_common_parent] / frequency
                    
                    extracted_patterns.append({
                        'pattern_type': pattern_type,
                        'pattern_text': pattern_text,
                        'target_parents': list(parent_counts.keys()),
                        'most_common_parent': most_common_parent,
                        'frequency': frequency,
                        'confidence': confidence,
                        'examples': [{'child': occ[0], 'parent': occ[1], 'relation': occ[2]} 
                                   for occ in occurrences[:3]]  # Limiter les exemples
                    })
                    
                    self.stats['patterns_extracted'] += 1
        
        # Trier par fréquence décroissante
        extracted_patterns.sort(key=lambda x: x['frequency'], reverse=True)
        
        self.extracted_patterns = extracted_patterns
        logger.info(f"Terminé. {len(extracted_patterns)} patterns extraits.")
        
        return extracted_patterns
    
    def apply_patterns_to_discover_relations(self, 
                                           patterns: List[Dict[str, Any]], 
                                           df: pd.DataFrame,
                                           existing_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applique les patterns extraits pour découvrir de nouvelles relations par analogie.
        
        Utilise les patterns relationnels pour identifier des termes qui pourraient
        avoir des relations similaires à celles observées dans les données existantes.
        
        Args:
            patterns (List[Dict[str, Any]]): Patterns extraits par extract_relation_patterns
            df (pd.DataFrame): DataFrame complet des termes
            existing_relations (List[Dict[str, Any]]): Relations existantes pour éviter doublons
            
        Returns:
            List[Dict[str, Any]]: Nouvelles relations découvertes par pattern matching
        """
        logger.info("Application des patterns pour découverte de relations...")
        
        discovered_relations = []
        existing_pairs = set()
        
        # Créer un set des paires existantes pour éviter les doublons
        for rel in existing_relations:
            child_uri = rel.get('child_uri')
            parent_uri = rel.get('parent_uri')
            if child_uri and parent_uri:
                existing_pairs.add((child_uri, parent_uri))
            else:
                existing_pairs.add((rel.get('child', ''), rel.get('parent', '')))
        
        # Appliquer chaque pattern
        for pattern in tqdm(patterns, desc="Application des patterns", unit="pattern"):
            pattern_type = pattern['pattern_type']
            pattern_text = pattern['pattern_text']
            target_parents = pattern['target_parents']
            confidence = pattern['confidence']
            
            # Rechercher des termes correspondant au pattern
            matching_terms = []
            
            for _, row in df.iterrows():
                term = row.get('preflabel_clean', '').strip()
                if not term:
                    continue
                    
                matches_pattern = False
                
                if pattern_type == 'suffix' and '*-' in pattern_text:
                    suffix = pattern_text.replace('*-', '')
                    if term.endswith(f'-{suffix}') or term.endswith(f' {suffix}'):
                        matches_pattern = True
                        
                elif pattern_type == 'prefix' and '* ' in pattern_text:
                    prefix = pattern_text.replace(' *', '')
                    if term.startswith(f'{prefix} ') or term.startswith(f'{prefix}-'):
                        matches_pattern = True
                        
                elif pattern_type == 'contains' and '*' in pattern_text:
                    keyword = pattern_text.replace('*', '')
                    if keyword.lower() in term.lower():
                        matches_pattern = True
                
                if matches_pattern:
                    # Vérifier que le terme n'a pas déjà une relation vers ces parents
                    term_uri = self.preflabel_to_uri_map.get(normalize_text(term))
                    matching_terms.append((term, term_uri))
            
            # Pour chaque terme correspondant, proposer des relations vers les parents cibles
            for term, term_uri in matching_terms:
                for target_parent in target_parents:
                    target_parent_uri = self.preflabel_to_uri_map.get(normalize_text(target_parent))
                    
                    # Vérifier que la relation n'existe pas déjà
                    relation_key = (term_uri, target_parent_uri) if term_uri and target_parent_uri else (term, target_parent)
                    
                    if relation_key not in existing_pairs:
                        # Calculer la similarité contextuelle pour validation
                        contextual_similarity = self.calculate_contextual_similarity(
                            term, target_parent, df, existing_relations
                        )
                        
                        if contextual_similarity >= self.analogy_similarity_threshold:
                            relation_type = 'existing_parent' if target_parent_uri else 'candidate_parent'
                            
                            discovered_relations.append({
                                'child': term,
                                'child_uri': term_uri,
                                'parent': target_parent,
                                'parent_uri': target_parent_uri,
                                'relation_category': 'pattern_analogy',
                                'type': relation_type,
                                'confidence': confidence * contextual_similarity,  # Pondérer par similarité contextuelle
                                'source': 'contextual_embedding',
                                'discovery_method': f'pattern_matching_{pattern_type}',
                                'contextual_similarity': contextual_similarity,
                                'supporting_evidence': {
                                    'pattern': pattern_text,
                                    'pattern_frequency': pattern['frequency'],
                                    'pattern_confidence': confidence,
                                    'examples': pattern['examples'][:2]  # Limiter les exemples
                                }
                            })
                            
                            existing_pairs.add(relation_key)  # Éviter les doublons dans cette session
                            self.stats['relations_discovered_by_pattern'] += 1
        
        logger.info(f"Terminé. {len(discovered_relations)} relations découvertes par patterns.")
        return discovered_relations
    
    def calculate_contextual_similarity(self, 
                                      term1: str, 
                                      term2: str, 
                                      df: pd.DataFrame, 
                                      all_relations: List[Dict[str, Any]]) -> float:
        """
        Calcule la similarité contextuelle entre deux termes basée sur leurs embeddings enrichis.
        
        Args:
            term1 (str): Premier terme à comparer
            term2 (str): Deuxième terme à comparer  
            df (pd.DataFrame): DataFrame des termes
            all_relations (List[Dict[str, Any]]): Relations pour enrichissement contextuel
            
        Returns:
            float: Score de similarité contextuelle normalisé entre 0.0 et 1.0
        """
        if not term1 or not term2:
            return 0.0
            
        emb1 = self.create_contextual_embedding(term1, df, all_relations)
        emb2 = self.create_contextual_embedding(term2, df, all_relations)
        
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        
        # Normaliser de -1,1 vers 0,1
        return (similarity + 1) / 2
    
    def build_relation_graph(self, 
                           df: pd.DataFrame, 
                           all_relations: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Construit un graphe dirigé des relations avec embeddings contextuels comme attributs.
        
        Crée une représentation graphique des relations existantes où :
        - Les nœuds sont les termes avec leurs embeddings contextuels
        - Les arêtes sont les relations avec leurs confidences comme poids
        
        Args:
            df (pd.DataFrame): DataFrame des termes
            all_relations (List[Dict[str, Any]]): Toutes les relations existantes
            
        Returns:
            nx.DiGraph: Graphe dirigé des relations avec attributs enrichis
        """
        logger.info("Construction du graphe de relations...")
        
        graph = nx.DiGraph()
        
        # Ajouter tous les termes comme nœuds avec leurs embeddings contextuels
        for _, row in df.iterrows():
            term = row.get('preflabel_clean', '').strip()
            if term:
                contextual_embedding = self.create_contextual_embedding(term, df, all_relations)
                
                graph.add_node(term, 
                             uri=row.get('URI'),
                             definition=row.get('definition_clean', ''),
                             source=row.get('source', ''),
                             contextual_embedding=contextual_embedding)
        
        # Ajouter les relations comme arêtes
        for relation in all_relations:
            child = relation.get('child', '').strip()
            parent = relation.get('parent', '').strip()
            confidence = relation.get('confidence', 0.5)
            
            if child and parent and child in graph.nodes and parent in graph.nodes:
                graph.add_edge(child, parent, 
                             weight=confidence,
                             relation_category=relation.get('relation_category', ''),
                             source=relation.get('source', ''))
        
        self.relation_graph = graph
        logger.info(f"Graphe construit avec {graph.number_of_nodes()} nœuds et {graph.number_of_edges()} arêtes.")
        
        return graph
    
    def discover_by_graph_proximity(self, 
                                  graph: nx.DiGraph, 
                                  df: pd.DataFrame,
                                  existing_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Découvre de nouvelles relations par analyse de proximité dans le graphe.
        
        Utilise la structure du graphe de relations pour identifier des connexions
        potentielles basées sur :
        - Proximité structurelle (chemins courts)
        - Similarité des voisinages
        - Patterns de connectivité
        
        Args:
            graph (nx.DiGraph): Graphe de relations construit
            df (pd.DataFrame): DataFrame des termes
            existing_relations (List[Dict[str, Any]]): Relations existantes
            
        Returns:
            List[Dict[str, Any]]: Nouvelles relations découvertes par proximité graphique
        """
        logger.info("Découverte par proximité graphique...")
        
        discovered_relations = []
        existing_pairs = set()
        
        # Créer un set des paires existantes
        for rel in existing_relations:
            existing_pairs.add((rel.get('child', ''), rel.get('parent', '')))
        
        # Analyser les chemins courts pour identifier des relations potentielles
        for node in graph.nodes():
            try:
                # Trouver les nœuds à distance 2 (grand-parents potentiels)
                nodes_at_distance_2 = []
                
                for target in graph.nodes():
                    if target != node and (node, target) not in existing_pairs:
                        try:
                            # Vérifier s'il existe un chemin de longueur 2
                            paths = list(nx.all_simple_paths(graph, node, target, cutoff=2))
                            if any(len(path) == 3 for path in paths):  # Chemin de longueur 2 (3 nœuds)
                                nodes_at_distance_2.append(target)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                
                # Pour chaque grand-parent potentiel, évaluer la pertinence
                for potential_parent in nodes_at_distance_2:
                    # Calculer la similarité contextuelle
                    contextual_similarity = self.calculate_contextual_similarity(
                        node, potential_parent, df, existing_relations
                    )
                    
                    if contextual_similarity >= self.analogy_similarity_threshold:
                        # Analyser le support structurel
                        common_neighbors = set(graph.successors(node)) & set(graph.successors(potential_parent))
                        structural_support = len(common_neighbors)
                        
                        if structural_support > 0:  # Au moins un parent en commun
                            potential_parent_uri = self.preflabel_to_uri_map.get(normalize_text(potential_parent))
                            node_uri = self.preflabel_to_uri_map.get(normalize_text(node))
                            relation_type = 'existing_parent' if potential_parent_uri else 'candidate_parent'
                            
                            confidence = contextual_similarity * (structural_support / max(graph.out_degree(node), 1))
                            
                            discovered_relations.append({
                                'child': node,
                                'child_uri': node_uri,
                                'parent': potential_parent,
                                'parent_uri': potential_parent_uri,
                                'relation_category': 'graph_proximity',
                                'type': relation_type,
                                'confidence': confidence,
                                'source': 'contextual_embedding',
                                'discovery_method': 'graph_proximity_analysis',
                                'contextual_similarity': contextual_similarity,
                                'supporting_evidence': {
                                    'structural_support': structural_support,
                                    'common_neighbors': list(common_neighbors)[:3],  # Limiter
                                    'graph_distance': 2
                                }
                            })
                            
                            existing_pairs.add((node, potential_parent))
                            self.stats['relations_discovered_by_graph'] += 1
                            
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse de proximité pour '{node}': {e}")
                continue
        
        logger.info(f"Terminé. {len(discovered_relations)} relations découvertes par proximité graphique.")
        return discovered_relations
    
    def identify_semantic_zones(self, 
                              df: pd.DataFrame, 
                              all_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifie des zones sémantiques cohérentes par clustering des embeddings contextuels.
        
        Utilise le clustering hiérarchique sur les embeddings contextuels pour identifier
        des groupes de termes sémantiquement cohérents qui forment des zones conceptuelles.
        
        Args:
            df (pd.DataFrame): DataFrame des termes
            all_relations (List[Dict[str, Any]]): Relations existantes pour contexte
            
        Returns:
            List[Dict[str, Any]]: Zones sémantiques identifiées avec leurs propriétés
        """
        logger.info("Identification des zones sémantiques...")
        
        terms = df['preflabel_clean'].dropna().tolist()
        if len(terms) < 2:
            logger.warning("Pas assez de termes pour identifier des zones sémantiques.")
            return []
        
        # Générer les embeddings contextuels pour tous les termes
        contextual_embeddings = []
        valid_terms = []
        
        for term in tqdm(terms, desc="Génération embeddings contextuels", unit="terme"):
            try:
                embedding = self.create_contextual_embedding(term, df, all_relations)
                contextual_embeddings.append(embedding)
                valid_terms.append(term)
            except Exception as e:
                logger.warning(f"Erreur embedding contextuel pour '{term}': {e}")
                continue
        
        if len(contextual_embeddings) < 2:
            logger.warning("Pas assez d'embeddings valides pour le clustering.")
            return []
        
        contextual_embeddings = np.array(contextual_embeddings)
        
        # Clustering hiérarchique adaptatif
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.zone_coherence_threshold,  # Convertir similarité en distance
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(contextual_embeddings)
        
        # Analyser les clusters pour former les zones sémantiques
        zones = defaultdict(list)
        for i, label in enumerate(labels):
            zones[label].append(valid_terms[i])
        
        semantic_zones = []
        for zone_id, zone_terms in zones.items():
            if len(zone_terms) >= 2:  # Zones avec au moins 2 termes
                # Calculer la cohérence interne de la zone
                zone_embeddings = np.array([
                    self.create_contextual_embedding(term, df, all_relations) 
                    for term in zone_terms
                ])
                
                internal_similarities = cosine_similarity(zone_embeddings)
                avg_internal_coherence = np.mean(internal_similarities[np.triu_indices(len(zone_embeddings), k=1)])
                avg_internal_coherence = (avg_internal_coherence + 1) / 2  # Normaliser
                
                # Identifier le terme le plus représentatif (centroïde)
                centroid = np.mean(zone_embeddings, axis=0)
                similarities_to_centroid = [
                    cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0]
                    for emb in zone_embeddings
                ]
                most_representative_idx = np.argmax(similarities_to_centroid)
                representative_term = zone_terms[most_representative_idx]
                
                # Analyser les sources pour identifier le domaine thématique
                zone_sources = []
                for term in zone_terms:
                    term_row = df[df['preflabel_clean'] == term]
                    if not term_row.empty:
                        source = term_row.iloc[0].get('source', '')
                        if source:
                            zone_sources.append(source)
                
                dominant_source = Counter(zone_sources).most_common(1)
                dominant_source = dominant_source[0][0] if dominant_source else None
                
                semantic_zones.append({
                    'zone_id': zone_id,
                    'terms': zone_terms,
                    'size': len(zone_terms),
                    'representative_term': representative_term,
                    'internal_coherence': avg_internal_coherence,
                    'dominant_source': dominant_source,
                    'source_diversity': len(set(zone_sources)) if zone_sources else 0
                })
                
                self.stats['semantic_zones_identified'] += 1
                self.stats['terms_in_zones'] += len(zone_terms)
        
        # Trier par cohérence décroissante
        semantic_zones.sort(key=lambda x: x['internal_coherence'], reverse=True)
        
        self.semantic_zones = semantic_zones
        logger.info(f"Terminé. {len(semantic_zones)} zones sémantiques identifiées.")
        
        return semantic_zones
    
    def enrich_zones_with_missing_relations(self, 
                                           zones: List[Dict[str, Any]], 
                                           df: pd.DataFrame,
                                           existing_relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Propose des relations manquantes au sein des zones sémantiques cohérentes.
        
        Pour chaque zone sémantique, identifie des relations hiérarchiques potentielles
        en utilisant le terme le plus représentatif comme parent candidat.
        
        Args:
            zones (List[Dict[str, Any]]): Zones sémantiques identifiées
            df (pd.DataFrame): DataFrame des termes
            existing_relations (List[Dict[str, Any]]): Relations existantes
            
        Returns:
            List[Dict[str, Any]]: Nouvelles relations découvertes par enrichissement de zones
        """
        logger.info("Enrichissement des zones sémantiques...")
        
        discovered_relations = []
        existing_pairs = set()
        
        # Créer un set des paires existantes
        for rel in existing_relations:
            existing_pairs.add((rel.get('child', ''), rel.get('parent', '')))
        
        for zone in zones:
            zone_terms = zone['terms']
            representative_term = zone['representative_term']
            internal_coherence = zone['internal_coherence']
            
            # Seulement pour les zones suffisamment cohérentes
            if internal_coherence >= self.zone_coherence_threshold:
                representative_uri = self.preflabel_to_uri_map.get(normalize_text(representative_term))
                
                # Proposer le terme représentatif comme parent pour les autres termes de la zone
                for term in zone_terms:
                    if term != representative_term and (term, representative_term) not in existing_pairs:
                        # Vérifier que cette relation apporte de la valeur
                        contextual_similarity = self.calculate_contextual_similarity(
                            term, representative_term, df, existing_relations
                        )
                        
                        if contextual_similarity >= self.family_similarity_threshold:
                            term_uri = self.preflabel_to_uri_map.get(normalize_text(term))
                            relation_type = 'existing_parent' if representative_uri else 'candidate_parent'
                            
                            discovered_relations.append({
                                'child': term,
                                'child_uri': term_uri,
                                'parent': representative_term,
                                'parent_uri': representative_uri,
                                'relation_category': 'semantic_zone_enrichment',
                                'type': relation_type,
                                'confidence': internal_coherence * contextual_similarity,
                                'source': 'contextual_embedding',
                                'discovery_method': 'zone_enrichment',
                                'contextual_similarity': contextual_similarity,
                                'supporting_evidence': {
                                    'zone_id': zone['zone_id'],
                                    'zone_coherence': internal_coherence,
                                    'zone_size': zone['size'],
                                    'dominant_source': zone.get('dominant_source')
                                }
                            })
                            
                            existing_pairs.add((term, representative_term))
                            self.stats['relations_discovered_by_zones'] += 1
        
        logger.info(f"Terminé. {len(discovered_relations)} relations découvertes par enrichissement de zones.")
        return discovered_relations
    
    def discover_abstract_concepts_from_zones(self, 
                                            zones: List[Dict[str, Any]], 
                                            df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifie des concepts abstraits manquants basés sur l'analyse des zones sémantiques.
        
        Pour les zones très cohérentes sans parent évident, propose des concepts
        abstraits générés automatiquement basés sur l'analyse terminologique.
        
        Args:
            zones (List[Dict[str, Any]]): Zones sémantiques identifiées
            df (pd.DataFrame): DataFrame des termes
            
        Returns:
            List[Dict[str, Any]]: Relations vers des concepts abstraits proposés
        """
        logger.info("Découverte de concepts abstraits à partir des zones sémantiques...")
        
        abstract_relations = []
        
        for zone in zones:
            zone_terms = zone['terms']
            internal_coherence = zone['internal_coherence']
            
            # Seulement pour les zones très cohérentes avec plusieurs termes
            if internal_coherence >= self.abstract_similarity_threshold and len(zone_terms) >= 3:
                # Analyser les mots récurrents pour générer un concept abstrait
                word_counter = Counter()
                for term in zone_terms:
                    # Nettoyer et compter les mots significatifs
                    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', term.lower())
                    word_counter.update(words)
                
                # Identifier le mot le plus fréquent comme base du concept abstrait
                if word_counter:
                    most_common_word, frequency = word_counter.most_common(1)[0]
                    
                    # Vérifier que ce concept n'existe pas déjà
                    abstract_concept = most_common_word.capitalize()
                    if abstract_concept.lower() not in df['preflabel_clean'].str.lower().values:
                        # Créer des relations vers ce concept abstrait
                        for term in zone_terms:
                            term_uri = self.preflabel_to_uri_map.get(normalize_text(term))
                            
                            abstract_relations.append({
                                'child': term,
                                'child_uri': term_uri,
                                'parent': abstract_concept,
                                'parent_uri': None,  # Sera généré en Phase 5
                                'relation_category': 'semantic_abstract_concept',
                                'type': 'candidate_parent',
                                'confidence': internal_coherence,
                                'source': 'contextual_embedding',
                                'discovery_method': 'zone_abstraction',
                                'contextual_similarity': internal_coherence,
                                'supporting_evidence': {
                                    'zone_id': zone['zone_id'],
                                    'zone_coherence': internal_coherence,
                                    'word_frequency': frequency,
                                    'zone_size': len(zone_terms),
                                    'generated_from': most_common_word
                                }
                            })
                            
                            self.stats['abstract_concepts_from_zones'] += 1
        
        logger.info(f"Terminé. {len(abstract_relations)} relations vers concepts abstraits découverts.")
        return abstract_relations
    
    def analyze_embeddings(self, 
                          df: pd.DataFrame, 
                          all_prev_relations: List[Dict[str, Any]],
                          lexical_indexes: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
        """
        Exécute l'analyse complète de découverte sémantique contextuelle.
        
        Cette méthode constitue le point d'entrée principal de la Phase 4 transformée.
        Elle effectue une découverte proactive de nouvelles relations en exploitant
        les relations existantes comme contexte d'apprentissage.
        
        Le processus comprend :
        
        1. **Extraction de patterns** : Identification des règles récurrentes dans
           les relations existantes
        2. **Construction du graphe** : Représentation structurelle des relations
        3. **Identification de zones** : Clustering sémantique contextuel
        4. **Découverte multi-méthodes** : 
           - Pattern matching analogique
           - Proximité graphique  
           - Enrichissement intra-zones
           - Concepts abstraits manquants
        
        Args:
            df (pd.DataFrame): DataFrame du thésaurus avec colonnes complètes
            all_prev_relations (List[Dict[str, Any]]): Toutes les relations des phases précédentes
            lexical_indexes (Dict[str, Any]): Index lexicaux avec mappings URI/termes
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]: Tuple contenant :
            
            - **nouvelles_relations** : Relations inédites découvertes (non-validation des existantes)
            - **zones_semantiques** : Zones conceptuelles identifiées avec statistiques
            - **contextes_enrichis** : Mapping terme → contexte textuel enrichi
        """
        logger.info("Début de la découverte sémantique contextuelle.")
        
        # Initialiser les mappings
        self.preflabel_to_uri_map = lexical_indexes.get('preflabel_to_uri', {})
        self.uri_to_preflabel_map = lexical_indexes.get('uri_to_preflabel', {})
        self.altlabel_to_uri_map = lexical_indexes.get('altlabel_to_uri', {})
        
        # 1. Extraction des patterns relationnels
        logger.info("Phase 1: Extraction des patterns relationnels")
        patterns = self.extract_relation_patterns(all_prev_relations)
        
        # 2. Construction du graphe de relations
        logger.info("Phase 2: Construction du graphe relationnel")
        relation_graph = self.build_relation_graph(df, all_prev_relations)
        
        # 3. Identification des zones sémantiques
        logger.info("Phase 3: Identification des zones sémantiques")
        semantic_zones = self.identify_semantic_zones(df, all_prev_relations)
        
        # 4. Découverte de nouvelles relations par différentes méthodes
        logger.info("Phase 4: Découverte multi-méthodes")
        
        # 4.1 Découverte par pattern matching
        pattern_relations = self.apply_patterns_to_discover_relations(patterns, df, all_prev_relations)
        
        # 4.2 Découverte par proximité graphique
        graph_relations = self.discover_by_graph_proximity(relation_graph, df, all_prev_relations)
        
        # 4.3 Enrichissement des zones sémantiques
        zone_relations = self.enrich_zones_with_missing_relations(semantic_zones, df, all_prev_relations)
        
        # 4.4 Concepts abstraits à partir des zones
        abstract_relations = self.discover_abstract_concepts_from_zones(semantic_zones, df)
        
        # 5. Consolidation et déduplication des relations découvertes
        logger.info("Phase 5: Consolidation des découvertes")
        all_discovered_relations = pattern_relations + graph_relations + zone_relations + abstract_relations
        
        # Déduplication basée sur (child, parent)
        seen_pairs = set()
        deduplicated_relations = []
        
        for relation in all_discovered_relations:
            pair_key = (relation.get('child', ''), relation.get('parent', ''))
            if pair_key not in seen_pairs:
                deduplicated_relations.append(relation)
                seen_pairs.add(pair_key)
            else:
                self.stats['duplicate_relations_removed'] += 1
        
        # 6. Construction du mapping des contextes enrichis
        contextes_enrichis = {}
        for term in df['preflabel_clean'].dropna():
            if term.strip():
                contextes_enrichis[term] = self.build_term_context(term, df, all_prev_relations)
        
        # Mise à jour des statistiques finales
        self.stats['total_relations_discovered'] = len(deduplicated_relations)
        self.stats['pattern_relations'] = len(pattern_relations)
        self.stats['graph_relations'] = len(graph_relations)
        self.stats['zone_relations'] = len(zone_relations)
        self.stats['abstract_relations'] = len(abstract_relations)
        
        logger.info(f"Découverte sémantique terminée. {len(deduplicated_relations)} nouvelles relations découvertes.")
        logger.info(f"  - Par pattern matching: {len(pattern_relations)}")
        logger.info(f"  - Par proximité graphique: {len(graph_relations)}")
        logger.info(f"  - Par enrichissement de zones: {len(zone_relations)}")
        logger.info(f"  - Concepts abstraits: {len(abstract_relations)}")
        logger.info(f"  - Zones sémantiques identifiées: {len(semantic_zones)}")
        
        return deduplicated_relations, semantic_zones, contextes_enrichis
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées de la découverte sémantique contextuelle.
        
        Fournit un aperçu complet des performances et résultats de la découverte,
        incluant les métriques par méthode et les statistiques de contextualisation.
        
        Returns:
            Dict[str, Any]: Dictionnaire contenant les statistiques suivantes :
            
            **Configuration** :
            
            - model_path (str): Chemin du modèle utilisé
            - context_window_size (int): Taille de la fenêtre contextuelle
            - pattern_min_frequency (int): Fréquence minimale des patterns
            - analogy_similarity_threshold (float): Seuil des analogies
            - zone_coherence_threshold (float): Seuil de cohérence des zones
            
            **Performance contextuelle** :
            
            - contexts_built (int): Contextes enrichis construits
            - contextual_embeddings_calculated (int): Nouveaux embeddings contextuels
            - contextual_embeddings_from_cache (int): Embeddings contextuels en cache
            
            **Extraction de patterns** :
            
            - patterns_extracted (int): Patterns relationnels extraits
            - relations_discovered_by_pattern (int): Relations par pattern matching
            
            **Analyse graphique** :
            
            - relations_discovered_by_graph (int): Relations par proximité graphique
            
            **Zones sémantiques** :
            
            - semantic_zones_identified (int): Zones sémantiques trouvées
            - terms_in_zones (int): Termes dans les zones
            - relations_discovered_by_zones (int): Relations par enrichissement de zones
            - abstract_concepts_from_zones (int): Concepts abstraits des zones
            
            **Totaux de découverte** :
            
            - total_relations_discovered (int): Total des nouvelles relations
            - pattern_relations (int): Relations par patterns
            - graph_relations (int): Relations par graphe
            - zone_relations (int): Relations par zones
            - abstract_relations (int): Relations abstraites
            - duplicate_relations_removed (int): Doublons supprimés
        """
        stats = dict(self.stats)
        stats.update({
            'model_path': self.model_path,
            'context_window_size': self.context_window_size,
            'pattern_min_frequency': self.pattern_min_frequency,
            'analogy_similarity_threshold': self.analogy_similarity_threshold,
            'zone_coherence_threshold': self.zone_coherence_threshold,
            'abstract_similarity_threshold': self.abstract_similarity_threshold,
            'family_similarity_threshold': self.family_similarity_threshold,
            'total_contextual_embeddings': stats.get('contextual_embeddings_calculated', 0) + 
                                         stats.get('contextual_embeddings_from_cache', 0),
        })
        return stats
    
    def export_discovery_network(self) -> Dict[str, Any]:
        """
        Exporte une représentation du réseau de découverte pour visualisation.
        
        Crée une structure adaptée pour visualiser les relations découvertes,
        les zones sémantiques et les patterns extraits.
        
        Returns:
            Dict[str, Any]: Structure contenant :
            
            - nodes (List[Dict]): Nœuds avec propriétés contextuelles
            - edges (List[Dict]): Arêtes avec méthodes de découverte
            - zones (List[Dict]): Zones sémantiques avec cohérence
            - patterns (List[Dict]): Patterns extraits avec fréquences
            - statistics (Dict): Statistiques du réseau de découverte
        """
        nodes = []
        edges = []
        
        # Construire les nœuds à partir des zones sémantiques
        node_id_map = {}
        node_counter = 0
        
        for zone in self.semantic_zones:
            for term in zone['terms']:
                if term not in node_id_map:
                    term_uri = self.preflabel_to_uri_map.get(term)
                    nodes.append({
                        'id': node_counter,
                        'label': term,
                        'uri': term_uri,
                        'zone_id': zone['zone_id'],
                        'zone_coherence': zone['internal_coherence'],
                        'is_representative': term == zone['representative_term']
                    })
                    node_id_map[term] = node_counter
                    node_counter += 1
        
        # Construire les arêtes à partir du graphe de relations
        if self.relation_graph:
            for source, target, data in self.relation_graph.edges(data=True):
                if source in node_id_map and target in node_id_map:
                    edges.append({
                        'source': node_id_map[source],
                        'target': node_id_map[target],
                        'weight': data.get('weight', 0.5),
                        'relation_category': data.get('relation_category', ''),
                        'source_method': data.get('source', '')
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'zones': self.semantic_zones,
            'patterns': self.extracted_patterns,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_zones': len(self.semantic_zones),
                'total_patterns': len(self.extracted_patterns),
                'average_zone_coherence': np.mean([z['internal_coherence'] for z in self.semantic_zones]) if self.semantic_zones else 0,
                'average_pattern_frequency': np.mean([p['frequency'] for p in self.extracted_patterns]) if self.extracted_patterns else 0
            }
        }
    
    def get_zone_details(self, zone_id: int) -> Optional[Dict[str, Any]]:
        """
        Retourne les détails d'une zone sémantique spécifique.
        
        Args:
            zone_id (int): Identifiant de la zone à analyser
            
        Returns:
            Optional[Dict[str, Any]]: Détails de la zone ou None si non trouvée
        """
        for zone in self.semantic_zones:
            if zone['zone_id'] == zone_id:
                return zone
        return None
    
    def get_pattern_details(self, pattern_text: str) -> Optional[Dict[str, Any]]:
        """
        Retourne les détails d'un pattern relationnel spécifique.
        
        Args:
            pattern_text (str): Texte du pattern à analyser
            
        Returns:
            Optional[Dict[str, Any]]: Détails du pattern ou None si non trouvé
        """
        for pattern in self.extracted_patterns:
            if pattern['pattern_text'] == pattern_text:
                return pattern
        return None
    
    def reset_caches(self) -> None:
        """
        Réinitialise tous les caches et structures internes.
        
        Vide les caches d'embeddings contextuels et remet à zéro les données
        calculées, forçant une recalculation complète lors du prochain appel.
        """
        self.contextual_embeddings_cache = {}
        self.relation_graph = nx.DiGraph()
        self.extracted_patterns = []
        self.semantic_zones = []
        self.stats = defaultdict(int)
        logger.info("Caches et structures internes réinitialisés")