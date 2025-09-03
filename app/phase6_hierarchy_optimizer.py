"""
Module d'optimisation de la hirarchie du thésaurus - Phase 6.

Ce module contient la classe principale pour l'optimisation des relations entre les
concept SKOS en vue de la génération des fichiers de sortie.
"""

import logging
import copy
import pandas as pd
from typing import Dict, List, Tuple, Any
import networkx as nx

class HierarchyOptimizer:
    """
    Phase 6 : Optimisation hiérarchique
    
    Cette classe optimise la hiérarchie construite en :
    1. Éliminant les redondances transitives
    2. Validant les poly-hiérarchies
    3. Détectant et résolvant les cycles

    Elle met également à jour le Dataframe avec les URIs des nouveaux concepts
    """
    
    def __init__(self, max_parents_per_node: int = 3, logger=None):
        """
        Initialise l'optimiseur hiérarchique.
        
        Args:
            max_parents_per_node: Nombre maximum de parents autorisés par terme
            logger: Logger pour les messages
        """
        self.max_parents_per_node = max_parents_per_node
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistiques de l'optimisation
        self.stats = {
            'transitive_relations_removed': 0,
            'cycles_detected': 0,
            'cycles_resolved': 0,
            'poly_hierarchies_truncated': 0,
            'nodes_processed': 0,
            'new_uris_added': 0,
            'relations_before_optimization': 0,
            'relations_after_optimization': 0
        }

    def optimize_hierarchy(self, 
                          df: pd.DataFrame,
                          hierarchy: Dict[str, Any], 
                          enriched_indexes: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimise la hiérarchie complète et met à jour le DataFrame avec les nouvelles URIs.
        
        Args:
            df: DataFrame avec une colonne 'URI'
            hierarchy: Structure hiérarchique à optimiser
            enriched_indexes: Index enrichis des concepts
            
        Returns:
            Tuple: (DataFrame mis à jour, hiérarchie optimisée, rapport d'optimisation)
        """
        self.logger.info("Début de l'optimisation hiérarchique")

        # Copier le DataFrame pour éviter les modifications in-place
        updated_df = df.copy()
    
        # Obtenir les URIs actuelles du DataFrame
        existing_uris = set(updated_df['URI'].tolist()) if 'URI' in updated_df.columns else set()
        
        # Copier la hiérarchie pour éviter les modifications in-place
        optimized_hierarchy = copy.deepcopy(hierarchy)
        nodes = optimized_hierarchy.get('nodes', {})
        
        self.stats['nodes_processed'] = len(nodes)
        self.stats['relations_before_optimization'] = self._count_relations(nodes)
        
        # Étape 1: Construire le graphe dirigé
        graph = self._build_directed_graph(nodes)
        
        # Étape 2: Détecter et résoudre les cycles
        self._detect_and_resolve_cycles(graph, nodes)
        
        # Étape 3: Éliminer les redondances transitives
        self._eliminate_transitive_redundancies(graph, nodes)
        
        # Étape 4: Valider les poly-hiérarchies
        self._validate_poly_hierarchies(nodes)
        
        # Étape 5: Reconstruire la structure optimisée
        self._rebuild_hierarchy_structure(nodes)
        
        self.stats['relations_after_optimization'] = self._count_relations(nodes)
        
        # Enrichir le DataFrame avec les nouvelles URIs
        updated_df = self._enrich_dataframe_with_new_uris(updated_df, existing_uris, enriched_indexes)

        # Générer le rapport
        optimization_report = self._generate_optimization_report()
        
        self.logger.info("Optimisation hiérarchique terminée")
        self.logger.info(f"Relations supprimées par transitivité: {self.stats['transitive_relations_removed']}")
        self.logger.info(f"Cycles détectés et résolus: {self.stats['cycles_resolved']}")
        self.logger.info(f"Nouvelles URIs ajoutées au DataFrame: {self.stats.get('new_uris_added', 0)}")

        return updated_df, optimized_hierarchy, optimization_report

    def _build_directed_graph(self, nodes: Dict[str, Any]) -> nx.DiGraph:
        """
        Construit un graphe dirigé NetworkX à partir de la hiérarchie.
        
        Args:
            nodes: Dictionnaire des nœuds de la hiérarchie
            
        Returns:
            Graphe dirigé NetworkX
        """
        graph = nx.DiGraph()
        
        for uri, node_data in nodes.items():
            # Ajouter le nœud avec ses métadonnées
            graph.add_node(uri, **node_data)
            
            # Ajouter les arêtes parent -> enfant
            parents = node_data.get('parents', [])
            for parent_uri in parents:
                graph.add_edge(parent_uri, uri)
                
        self.logger.debug(f"Graphe construit avec {graph.number_of_nodes()} nœuds et {graph.number_of_edges()} arêtes")
        return graph

    def _detect_and_resolve_cycles(self, graph: nx.DiGraph, nodes: Dict[str, Any]) -> None:
        """
        Détecte et résout les cycles dans la hiérarchie.
        
        Args:
            graph: Graphe dirigé NetworkX
            nodes: Dictionnaire des nœuds de la hiérarchie
        """
        try:
            # Détecter les cycles
            cycles = list(nx.simple_cycles(graph))
            self.stats['cycles_detected'] = len(cycles)
            
            if cycles:
                self.logger.warning(f"Détection de {len(cycles)} cycle(s) dans la hiérarchie")
                
                for i, cycle in enumerate(cycles):
                    self.logger.warning(f"Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
                    
                    # Résoudre le cycle en supprimant l'arête la plus faible
                    self._resolve_cycle(cycle, graph, nodes)
                    self.stats['cycles_resolved'] += 1
                    
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection des cycles: {e}")

    def _resolve_cycle(self, cycle: List[str], graph: nx.DiGraph, nodes: Dict[str, Any]) -> None:
        """
        Résout un cycle en supprimant l'arête avec le score de confiance le plus faible.
        
        Args:
            cycle: Liste des URIs formant le cycle
            graph: Graphe dirigé NetworkX
            nodes: Dictionnaire des nœuds de la hiérarchie
        """
        if len(cycle) < 2:
            return
            
        # Trouver l'arête la plus faible dans le cycle
        weakest_edge = None
        min_confidence = float('inf')
        
        for i in range(len(cycle)):
            current_uri = cycle[i]
            next_uri = cycle[(i + 1) % len(cycle)]
            
            # Chercher la confiance de cette relation
            confidence = self._get_relation_confidence(current_uri, next_uri, nodes)
            
            if confidence < min_confidence:
                min_confidence = confidence
                weakest_edge = (current_uri, next_uri)
        
        if weakest_edge:
            parent_uri, child_uri = weakest_edge
            
            # Supprimer l'arête du graphe
            if graph.has_edge(parent_uri, child_uri):
                graph.remove_edge(parent_uri, child_uri)
            
            # Mettre à jour la structure des nœuds
            if child_uri in nodes and 'parents' in nodes[child_uri]:
                if parent_uri in nodes[child_uri]['parents']:
                    nodes[child_uri]['parents'].remove(parent_uri)
                    
            if parent_uri in nodes and 'children' in nodes[parent_uri]:
                if child_uri in nodes[parent_uri]['children']:
                    nodes[parent_uri]['children'].remove(child_uri)
            
            self.logger.info(f"Cycle résolu: suppression de la relation {parent_uri} -> {child_uri} (confiance: {min_confidence:.3f})")

    def _get_relation_confidence(self, parent_uri: str, child_uri: str, nodes: Dict[str, Any]) -> float:
        """
        Récupère le score de confiance d'une relation parent-enfant.
        
        Args:
            parent_uri: URI du parent
            child_uri: URI de l'enfant
            nodes: Dictionnaire des nœuds
            
        Returns:
            Score de confiance (0.0 si non trouvé)
        """
        # Chercher dans les métadonnées du nœud enfant
        child_node = nodes.get(child_uri, {})
        metadata = child_node.get('metadata', {})
        
        # Si on a des relations candidates avec des scores
        if 'relation_scores' in metadata:
            return metadata['relation_scores'].get(parent_uri, 0.5)
        
        # Score par défaut basé sur le type de source
        source = metadata.get('source', 'unknown')
        default_scores = {
            'existing_skos_broader': 0.9,
            'pattern_inclusion': 0.8,
            'lexical_similarity': 0.6,
            'embedding': 0.7,
            'generation': 0.5
        }
        
        return default_scores.get(source, 0.5)

    def _eliminate_transitive_redundancies(self, graph: nx.DiGraph, nodes: Dict[str, Any]) -> None:
        """
        Élimine les redondances transitives (A->B, B->C, A->C => supprimer A->C).
        
        Args:
            graph: Graphe dirigé NetworkX
            nodes: Dictionnaire des nœuds de la hiérarchie
        """
        self.logger.info("Élimination des redondances transitives")
        
        transitive_edges_to_remove = set()
        
        # Pour chaque nœud, vérifier les chemins transitifs
        for node_uri in list(graph.nodes()):
            # Obtenir tous les descendants directs et indirects
            direct_children = set(graph.successors(node_uri))
            
            if len(direct_children) < 2:
                continue
                
            # Calculer la fermeture transitive pour ce nœud
            reachable_nodes = set()
            for child in direct_children:
                # Obtenir tous les descendants de cet enfant direct
                descendants = nx.descendants(graph, child)
                reachable_nodes.update(descendants)
            
            # Identifier les arêtes redondantes
            redundant_edges = direct_children.intersection(reachable_nodes)
            
            for redundant_child in redundant_edges:
                # Vérifier qu'il existe bien un chemin alternatif
                if self._has_alternative_path(graph, node_uri, redundant_child, exclude_direct=True):
                    transitive_edges_to_remove.add((node_uri, redundant_child))
        
        # Supprimer les arêtes transitives identifiées
        for parent_uri, child_uri in transitive_edges_to_remove:
            if graph.has_edge(parent_uri, child_uri):
                graph.remove_edge(parent_uri, child_uri)
                
                # Mettre à jour la structure des nœuds
                if child_uri in nodes and 'parents' in nodes[child_uri]:
                    if parent_uri in nodes[child_uri]['parents']:
                        nodes[child_uri]['parents'].remove(parent_uri)
                        
                if parent_uri in nodes and 'children' in nodes[parent_uri]:
                    if child_uri in nodes[parent_uri]['children']:
                        nodes[parent_uri]['children'].remove(child_uri)
                
                self.stats['transitive_relations_removed'] += 1
                self.logger.debug(f"Relation transitive supprimée: {parent_uri} -> {child_uri}")

    def _has_alternative_path(self, graph: nx.DiGraph, source: str, target: str, exclude_direct: bool = True) -> bool:
        """
        Vérifie s'il existe un chemin alternatif entre deux nœuds.
        
        Args:
            graph: Graphe dirigé
            source: Nœud source
            target: Nœud cible
            exclude_direct: Si True, ignore le chemin direct
            
        Returns:
            True s'il existe un chemin alternatif
        """
        if not graph.has_node(source) or not graph.has_node(target):
            return False
        
        # Créer une copie temporaire du graphe
        temp_graph = graph.copy()
        
        # Supprimer l'arête directe si elle existe et si demandé
        if exclude_direct and temp_graph.has_edge(source, target):
            temp_graph.remove_edge(source, target)
        
        # Vérifier s'il existe encore un chemin
        try:
            return nx.has_path(temp_graph, source, target)
        except nx.NetworkXNoPath:
            return False

    def _validate_poly_hierarchies(self, nodes: Dict[str, Any]) -> None:
        """
        Valide les poly-hiérarchies en limitant le nombre de parents par nœud.
        
        Args:
            nodes: Dictionnaire des nœuds de la hiérarchie
        """
        self.logger.info(f"Validation des poly-hiérarchies (max {self.max_parents_per_node} parents)")
        
        for uri, node_data in nodes.items():
            parents = node_data.get('parents', [])
            
            if len(parents) > self.max_parents_per_node:
                self.logger.warning(f"Nœud {uri} a {len(parents)} parents, réduction à {self.max_parents_per_node}")
                
                # Trier les parents par score de confiance (décroissant)
                parent_scores = []
                for parent_uri in parents:
                    confidence = self._get_relation_confidence(parent_uri, uri, nodes)
                    parent_scores.append((parent_uri, confidence))
                
                # Garder seulement les N meilleurs parents
                parent_scores.sort(key=lambda x: x[1], reverse=True)
                best_parents = [parent for parent, _ in parent_scores[:self.max_parents_per_node]]
                removed_parents = [parent for parent, _ in parent_scores[self.max_parents_per_node:]]
                
                # Mettre à jour la liste des parents
                node_data['parents'] = best_parents
                
                # Mettre à jour les enfants des parents supprimés
                for removed_parent in removed_parents:
                    if removed_parent in nodes and 'children' in nodes[removed_parent]:
                        if uri in nodes[removed_parent]['children']:
                            nodes[removed_parent]['children'].remove(uri)
                
                self.stats['poly_hierarchies_truncated'] += 1
                self.logger.debug(f"Parents supprimés pour {uri}: {removed_parents}")

    def _rebuild_hierarchy_structure(self, nodes: Dict[str, Any]) -> None:
        """
        Reconstruit la cohérence de la structure hiérarchique.
        
        Args:
            nodes: Dictionnaire des nœuds de la hiérarchie
        """
        self.logger.debug("Reconstruction de la structure hiérarchique")
        
        # Vérifier la cohérence parent-enfant
        for uri, node_data in nodes.items():
            parents = node_data.get('parents', [])
            children = node_data.get('children', [])
            
            # S'assurer que tous les parents connaissent cet enfant
            for parent_uri in parents:
                if parent_uri in nodes:
                    parent_children = nodes[parent_uri].setdefault('children', [])
                    if uri not in parent_children:
                        parent_children.append(uri)
            
            # S'assurer que tous les enfants connaissent ce parent
            for child_uri in children:
                if child_uri in nodes:
                    child_parents = nodes[child_uri].setdefault('parents', [])
                    if uri not in child_parents:
                        child_parents.append(uri)

    def _enrich_dataframe_with_new_uris(self, df: Any, existing_uris: set, enriched_indexes: Dict[str, Any]) -> Any:
        """
        Enrichit le DataFrame avec les URIs présentes dans enriched_indexes mais absentes du DataFrame.
        
        Args:
            df: DataFrame à enrichir
            existing_uris: Set des URIs déjà présentes dans le DataFrame
            enriched_indexes: Index enrichis contenant potentiellement de nouvelles URIs
            
        Returns:
            DataFrame enrichi
        """
        new_rows = []
        new_uris_added = 0
        
        # Extraire les URIs des enriched_indexes
        enriched_uris = set()
        
        # Si enriched_indexes contient une structure avec des URIs
        if isinstance(enriched_indexes, dict):
            # Cas 1: enriched_indexes est un dict avec des clés URI
            enriched_uris.update(enriched_indexes.keys())
            
            # Cas 2: enriched_indexes contient des sous-structures avec des URIs
            for key, value in enriched_indexes.items():
                if isinstance(value, dict):
                    enriched_uris.update(value.keys())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            enriched_uris.add(item)
                        elif isinstance(item, dict) and 'uri' in item:
                            enriched_uris.add(item['uri'])
        
        # Trouver les URIs manquantes
        missing_uris = enriched_uris - existing_uris
        
        # Créer de nouvelles lignes pour les URIs manquantes
        for uri in missing_uris:
            # Créer une nouvelle ligne avec l'URI et des valeurs par défaut pour les autres colonnes
            new_row = {'URI': uri}
            
            # Ajouter des valeurs par défaut pour les autres colonnes existantes
            for col in df.columns:
                if col != 'URI':
                    new_row[col] = None  # ou une valeur par défaut appropriée
            
            new_rows.append(new_row)
            new_uris_added += 1
        
        if new_rows:
            # Créer un DataFrame avec les nouvelles lignes
            import pandas as pd  # Assurez-vous d'importer pandas en haut du fichier
            new_df = pd.DataFrame(new_rows)
            
            # Concaténer avec le DataFrame existant
            enriched_df = pd.concat([df, new_df], ignore_index=True)
            
            self.logger.info(f"Ajout de {new_uris_added} nouvelles URIs au DataFrame")
            self.stats['new_uris_added'] = new_uris_added
            
            return enriched_df
        else:
            self.logger.info("Aucune nouvelle URI à ajouter au DataFrame")
            self.stats['new_uris_added'] = 0
            return df

    def _count_relations(self, nodes: Dict[str, Any]) -> int:
        """
        Compte le nombre total de relations dans la hiérarchie.
        
        Args:
            nodes: Dictionnaire des nœuds
            
        Returns:
            Nombre total de relations parent-enfant
        """
        total_relations = 0
        for node_data in nodes.values():
            total_relations += len(node_data.get('children', []))
        return total_relations

    def _generate_optimization_report(self) -> Dict[str, Any]:
        """
        Génère un rapport détaillé de l'optimisation.
        
        Returns:
            Dictionnaire contenant les statistiques et métriques
        """
        relations_reduction = self.stats['relations_before_optimization'] - self.stats['relations_after_optimization']
        reduction_percentage = (relations_reduction / max(1, self.stats['relations_before_optimization'])) * 100
        
        report = {
            'optimization_summary': {
                'nodes_processed': self.stats['nodes_processed'],
                'relations_before': self.stats['relations_before_optimization'],
                'relations_after': self.stats['relations_after_optimization'],
                'relations_removed': relations_reduction,
                'reduction_percentage': round(reduction_percentage, 2)
            },
            'transitive_optimization': {
                'transitive_relations_removed': self.stats['transitive_relations_removed']
            },
            'cycle_resolution': {
                'cycles_detected': self.stats['cycles_detected'],
                'cycles_resolved': self.stats['cycles_resolved']
            },
            'poly_hierarchy_validation': {
                'max_parents_allowed': self.max_parents_per_node,
                'nodes_with_excess_parents': self.stats['poly_hierarchies_truncated']
            },
            'dataframe_enrichment': {
                'new_uris_added': self.stats.get('new_uris_added', 0),
                'enrichment_status': 'Success' if self.stats.get('new_uris_added', 0) >= 0 else 'Failed'
            },
            'quality_metrics': {
                'optimization_efficiency': round(self.stats['transitive_relations_removed'] / max(1, self.stats['relations_before_optimization']) * 100, 2),
                'hierarchy_health': 'Good' if self.stats['cycles_resolved'] == self.stats['cycles_detected'] else 'Needs attention',
                'dataframe_completeness': round((self.stats.get('new_uris_added', 0) / max(1, self.stats['nodes_processed'])) * 100, 2)
            }
        }
        
        return report

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de l'optimisation.
        
        Returns:
            Dictionnaire des statistiques
        """
        return self.stats.copy()

    def validate_hierarchy_integrity(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide l'intégrité de la hiérarchie après optimisation.
        
        Args:
            hierarchy: Structure hiérarchique à valider
            
        Returns:
            Rapport de validation
        """
        nodes = hierarchy.get('nodes', {})
        validation_report = {
            'total_nodes': len(nodes),
            'orphan_nodes': 0,
            'root_nodes': 0,
            'leaf_nodes': 0,
            'integrity_issues': [],
            'max_depth': 0,
            'average_children_per_node': 0,
            'average_parents_per_node': 0
        }
        
        total_children = 0
        total_parents = 0
        
        for uri, node_data in nodes.items():
            parents = node_data.get('parents', [])
            children = node_data.get('children', [])
            
            total_parents += len(parents)
            total_children += len(children)
            
            # Compter les différents types de nœuds
            if not parents and not children:
                validation_report['orphan_nodes'] += 1
            elif not parents:
                validation_report['root_nodes'] += 1
            elif not children:
                validation_report['leaf_nodes'] += 1
            
            # Vérifier l'intégrité des relations
            for parent_uri in parents:
                if parent_uri not in nodes:
                    validation_report['integrity_issues'].append(f"Parent manquant: {parent_uri} pour {uri}")
                elif uri not in nodes[parent_uri].get('children', []):
                    validation_report['integrity_issues'].append(f"Incohérence parent-enfant: {parent_uri} -> {uri}")
            
            for child_uri in children:
                if child_uri not in nodes:
                    validation_report['integrity_issues'].append(f"Enfant manquant: {child_uri} pour {uri}")
                elif uri not in nodes[child_uri].get('parents', []):
                    validation_report['integrity_issues'].append(f"Incohérence enfant-parent: {uri} -> {child_uri}")
        
        # Calculer les moyennes
        if validation_report['total_nodes'] > 0:
            validation_report['average_children_per_node'] = round(total_children / validation_report['total_nodes'], 2)
            validation_report['average_parents_per_node'] = round(total_parents / validation_report['total_nodes'], 2)
        
        # Calculer la profondeur maximale
        try:
            graph = self._build_directed_graph(nodes)
            if graph.number_of_nodes() > 0:
                # Trouver tous les nœuds racines
                root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
                max_depth = 0
                for root in root_nodes:
                    try:
                        depths = nx.single_source_shortest_path_length(graph, root)
                        max_depth = max(max_depth, max(depths.values()) if depths else 0)
                    except:
                        continue
                validation_report['max_depth'] = max_depth
        except Exception as e:
            validation_report['integrity_issues'].append(f"Erreur de calcul de profondeur: {str(e)}")
        
        return validation_report