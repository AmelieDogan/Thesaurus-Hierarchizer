"""
Module hierarchy_optimizer.py
Phase 6 : Optimisation de la hiérarchie.
"""

from typing import Dict, List, Any
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from operator import itemgetter

from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class OptimizationStats:
    """Statistiques d'optimisation pour le logging et le monitoring."""
    initial_relations_count: int = 0
    final_relations_count: int = 0
    redundancies_removed: int = 0
    cycles_detected: int = 0
    cycles_resolved: int = 0
    poly_hierarchy_adjustments: int = 0
    execution_time: float = 0.0

class HierarchyOptimizer:
    def __init__(self,
                 max_parents_per_term: int = 3,
                 enable_poly_hierarchy: bool = True):
        """
        Initialise l'optimiseur hiérarchique.

        Args:
            max_parents_per_term: Nombre maximum de parents autorisés par terme
            enable_poly_hierarchy: Autorise ou non plusieurs parents par terme
        """
        self.max_parents_per_term = max_parents_per_term
        self.enable_poly_hierarchy = enable_poly_hierarchy
        self.stats = OptimizationStats()
        
        logger.info("Optimiseur hiérarchique initialisé")
        logger.info(f"  - Max parents par terme : {max_parents_per_term}")
        logger.info(f"  - Poly-hiérarchie activée : {enable_poly_hierarchy}")

    def build_hierarchy_graph(self, relations: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Construit un graphe de la hiérarchie en utilisant les URIs.

        Args:
            relations: Liste des relations, chaque relation doit contenir 'child_uri' et 'parent_uri'.

        Returns:
            Un dictionnaire représentant le graphe d'adjacence {parent_uri: [child_uri]}.
        """
        graph = defaultdict(list)
        for rel in relations:
            # Assurez-vous que les URIs sont présentes. Si non, loggez une erreur et utilisez les prefLabels comme fallback.
            child_node = rel.get('child_uri')
            parent_node = rel.get('parent_uri')

            if not child_node:
                logger.warning(f"Relation sans 'child_uri', fallback sur 'child': {rel}")
                child_node = rel['child']
            if not parent_node:
                logger.warning(f"Relation sans 'parent_uri', fallback sur 'parent': {rel}")
                parent_node = rel['parent']
            
            # Éviter les boucles réflexives (terme parent de lui-même)
            if child_node != parent_node:
                graph[parent_node].append(child_node)
        return graph

    def get_paths(self, graph: Dict[str, List[str]], start_node: str, end_node: str) -> List[List[str]]:
        """
        Trouve tous les chemins entre deux nœuds dans le graphe en utilisant les URIs.
        
        Args:
            graph: Le graphe d'adjacence.
            start_node: L'URI du nœud de départ.
            end_node: L'URI du nœud d'arrivée.
            
        Returns:
            Liste de listes, chaque sous-liste étant un chemin (liste d'URIs).
        """
        paths = []
        queue = deque([(start_node, [start_node])]) # (current_node, current_path)
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node == end_node:
                paths.append(path)
                continue # Ne pas explorer plus loin pour ce chemin complet
            
            # Éviter les boucles dans un même chemin
            for neighbor in graph.get(current_node, []):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
        return paths

    def remove_redundancies(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Supprime les relations redondantes (transitives) en utilisant les URIs.
        Ex: Si A -> B et B -> C existent, et A -> C existe, alors A -> C est redondant.

        Args:
            relations: Liste des relations à optimiser.

        Returns:
            Liste des relations après suppression des redondances.
        """
        logger.info("Détection et suppression des redondances transitives...")
        initial_count = len(relations)
        
        # Créer un set de relations pour un lookup rapide et éviter les doublons accidentels
        # Une relation est identifiée par (child_uri, parent_uri)
        unique_relations = {} # {(child_uri, parent_uri): relation_dict}
        for rel in relations:
            child_uri = rel.get('child_uri')
            parent_uri = rel.get('parent_uri')
            if child_uri and parent_uri and child_uri != parent_uri:
                # Utiliser la confiance comme critère de choix si plusieurs relations identiques
                key = (child_uri, parent_uri)
                if key not in unique_relations or rel.get('confidence', 0) > unique_relations[key].get('confidence', 0):
                    unique_relations[key] = rel
            else:
                # Fallback ou ignorer les relations sans URIs valides pour la déduplication stricte
                logger.warning(f"Ignoré une relation sans URIs complètes pour la déduplication: {rel}")
        
        relations_to_process = list(unique_relations.values())
        
        # Construire le graphe avec les relations initiales
        graph = defaultdict(list)
        # Chaque nœud du graphe est un URI. Les arêtes vont de parent à enfant.
        for rel in relations_to_process:
            graph[rel['parent_uri']].append(rel['child_uri'])
            
        final_relations = []
        removed_count = 0
        
        for rel in relations_to_process:
            child_uri = rel['child_uri']
            parent_uri = rel['parent_uri']
            
            if child_uri == parent_uri: # Ignorer les relations réflexives
                removed_count += 1
                continue

            is_redundant = False
            
            # Vérifier si un chemin existe de parent_uri à child_uri en passant par au moins un autre nœud
            # Pour cela, on retire temporairement la relation directe (parent_uri -> child_uri)
            # pour voir si un chemin indirect existe toujours.
            
            # Supprimer temporairement l'arête directe
            if child_uri in graph[parent_uri]:
                graph[parent_uri].remove(child_uri)
            
            # Chercher un chemin de parent_uri à child_uri dans le graphe modifié
            # Une simple BFS/DFS suffit pour vérifier l'existence d'un chemin
            q = deque([parent_uri])
            visited = {parent_uri}
            
            while q:
                current_node = q.popleft()
                if current_node == child_uri:
                    is_redundant = True
                    break
                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            
            # Réajouter l'arête pour les itérations suivantes
            graph[parent_uri].append(child_uri)
            
            if is_redundant:
                removed_count += 1
            else:
                final_relations.append(rel)
        
        self.stats.redundancies_removed += removed_count
        logger.info(f"Supprimé {removed_count} relations redondantes. Reste {len(final_relations)} relations.")
        return final_relations

    def detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """
        Détecte les cycles dans le graphe de hiérarchie en utilisant les URIs (DFS).

        Args:
            graph: Le graphe d'adjacence {parent_uri: [child_uri]}.

        Returns:
            Liste de listes, chaque sous-liste étant un cycle (liste d'URIs).
        """
        logger.info("Détection des cycles dans la hiérarchie...")
        cycles = []
        visited = set()      # Nœuds visités
        recursion_stack = set() # Nœuds actuellement dans le chemin DFS
        
        # Pour stocker les chemins jusqu'au nœud courant
        path_tracker = {} 

        def dfs(node: str, current_path: List[str]):
            visited.add(node)
            recursion_stack.add(node)
            path_tracker[node] = current_path
            
            for neighbor in graph.get(node, []):
                if neighbor in recursion_stack: # Cycle détecté
                    cycle = current_path[current_path.index(neighbor):] + [neighbor]
                    if cycle not in cycles: # Éviter les doublons de cycles (ex: [A,B,C,A] vs [B,C,A,B])
                        # Normaliser le cycle pour le rendre unique (ex: commencer par le min URI)
                        min_uri = min(cycle[:-1])
                        start_idx = cycle[:-1].index(min_uri)
                        normalized_cycle = cycle[start_idx:-1] + cycle[:start_idx] + [min_uri] # Le dernier est le premier
                        
                        if normalized_cycle not in cycles:
                            cycles.append(normalized_cycle)
                elif neighbor not in visited:
                    dfs(neighbor, current_path + [neighbor])
            
            recursion_stack.remove(node) # Retirer du stack après exploration
            
        for node in graph: # Parcourir tous les nœuds du graphe
            if node not in visited:
                dfs(node, [node])
        
        self.stats.cycles_detected += len(cycles)
        logger.info(f"Détecté {len(cycles)} cycles.")
        return cycles

    def resolve_cycles(self, relations: List[Dict[str, Any]], cycles: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Résout les cycles en supprimant la relation la moins "confiante" dans chaque cycle.
        
        Args:
            relations: Liste des relations hiérarchiques.
            cycles: Liste des cycles détectés (chaque cycle est une liste d'URIs).
            
        Returns:
            Liste des relations après résolution des cycles.
        """
        logger.info(f"Résolution de {len(cycles)} cycles...")
        if not cycles:
            return relations

        relations_dict = {} # (child_uri, parent_uri) -> relation_dict
        for rel in relations:
            relations_dict[(rel['child_uri'], rel['parent_uri'])] = rel
        
        removed_relations_count = 0
        
        for cycle in cycles:
            # Un cycle est [A, B, C, A] -> relations sont (A,B), (B,C), (C,A)
            # Construire les paires (child_uri, parent_uri) pour chaque relation du cycle
            cycle_relations_keys = []
            for i in range(len(cycle) - 1):
                child_uri_in_cycle = cycle[i] # L'enfant dans le cycle
                parent_uri_in_cycle = cycle[i+1] # Le parent dans le cycle
                # Les relations dans notre modèle sont (child, parent), donc il faut inverser si le cycle est parent -> child -> ...
                # Si le cycle est A -> B -> C -> A, cela veut dire que A est parent de B, B parent de C, C parent de A.
                # Donc les relations sont (B,A), (C,B), (A,C).
                # Vérifions la structure du graphe: {parent_uri: [child_uri]}
                # Si cycle est [A, B, C, A], cela signifie A est enfant de C, B est enfant de A, C est enfant de B.
                # Ou si le graphe est {child_uri: [parent_uri]}, alors c'est A parent de B, B parent de C, C parent de A.
                # Le build_hierarchy_graph est {parent_uri: [child_uri]}, donc A -> B signifie (B, A)
                # Un cycle [N1, N2, ..., Nk, N1] signifie N1 parent de N2, N2 parent de N3, ..., Nk parent de N1
                # Donc les relations correspondantes sont (N2, N1), (N3, N2), ..., (N1, Nk)
                
                # Le cycle est de la forme [N1, N2, N3, N1] où N1 -> N2, N2 -> N3, N3 -> N1
                # Cela signifie les relations sont (N2, N1), (N3, N2), (N1, N3)
                child_uri = cycle[i+1] # L'enfant dans la relation
                parent_uri = cycle[i] # Le parent dans la relation
                cycle_relations_keys.append((child_uri, parent_uri))
            
            # La dernière relation du cycle est de cycle[-1] vers cycle[0]
            cycle_relations_keys.append((cycle[0], cycle[-2])) # Cycle est [N1, N2, N3, N1], rels sont (N2,N1), (N3,N2), (N1,N3)

            # Trouver la relation avec la plus faible confiance dans le cycle
            # Assurez-vous que toutes les clés existent dans relations_dict
            valid_cycle_relations = []
            for key in cycle_relations_keys:
                if key in relations_dict:
                    valid_cycle_relations.append(relations_dict[key])
                else:
                    logger.warning(f"Relation de cycle non trouvée dans le dictionnaire des relations : {key}")
            
            if not valid_cycle_relations:
                logger.warning(f"Aucune relation valide trouvée pour le cycle : {cycle}. Impossible de résoudre.")
                continue

            # Trouver la relation avec la plus faible confiance
            weakest_relation = min(valid_cycle_relations, key=itemgetter('confidence'))
            
            # Supprimer cette relation de la liste des relations
            key_to_remove = (weakest_relation['child_uri'], weakest_relation['parent_uri'])
            if key_to_remove in relations_dict:
                del relations_dict[key_to_remove]
                removed_relations_count += 1
                self.stats.cycles_resolved += 1
                logger.info(f"Résolu cycle en supprimant la relation: {weakest_relation['child']} -> {weakest_relation['parent']} (confiance: {weakest_relation['confidence']:.2f})")
            else:
                logger.warning(f"Tentative de suppression d'une relation déjà absente ou non trouvée lors de la résolution de cycle: {key_to_remove}")

        logger.info(f"Terminé. {removed_relations_count} relations supprimées pour résoudre les cycles.")
        return list(relations_dict.values())

    def validate_poly_hierarchy(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Valide et ajuste la poly-hiérarchie en limitant le nombre de parents par terme.
        Si la poly-hiérarchie est désactivée, s'assure qu'il n'y a qu'un seul parent.

        Args:
            relations: Liste des relations hiérarchiques.

        Returns:
            Liste des relations ajustées.
        """
        logger.info("Validation et ajustement de la poly-hiérarchie...")
        final_relations = []
        child_parent_counts = defaultdict(list) # child_uri -> [(parent_uri, confidence, relation_dict)]
        
        for rel in relations:
            child_uri = rel.get('child_uri')
            parent_uri = rel.get('parent_uri')
            if child_uri and parent_uri and child_uri != parent_uri:
                child_parent_counts[child_uri].append((parent_uri, rel.get('confidence', 0.0), rel))
            else:
                logger.warning(f"Ignoré une relation mal formée pour la validation de poly-hiérarchie: {rel}")

        adjustments_made = 0
        for child_uri, parents_info in child_parent_counts.items():
            if not self.enable_poly_hierarchy:
                # Si la poly-hiérarchie est désactivée, limiter à 1 parent
                if len(parents_info) > 1:
                    # Garder le parent avec la plus haute confiance
                    best_parent = max(parents_info, key=itemgetter(1)) # itemgetter(1) pour la confiance
                    final_relations.append(best_parent[2]) # Ajouter le dictionnaire de relation complet
                    adjustments_made += (len(parents_info) - 1)
                elif len(parents_info) == 1:
                    final_relations.append(parents_info[0][2])
            else:
                # Si la poly-hiérarchie est activée, mais limitée par max_parents_per_term
                if len(parents_info) > self.max_parents_per_term:
                    # Garder les N parents avec la plus haute confiance
                    sorted_parents = sorted(parents_info, key=itemgetter(1), reverse=True)
                    for i in range(self.max_parents_per_term):
                        final_relations.append(sorted_parents[i][2])
                    adjustments_made += (len(parents_info) - self.max_parents_per_term)
                else:
                    # Ajouter toutes les relations si elles respectent la limite
                    for parent_info in parents_info:
                        final_relations.append(parent_info[2])
                        
        self.stats.poly_hierarchy_adjustments += adjustments_made
        logger.info(f"Ajusté {adjustments_made} relations pour la poly-hiérarchie. Reste {len(final_relations)} relations.")
        return final_relations
        
    def optimize_hierarchy(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Exécute toutes les étapes d'optimisation hiérarchique.

        Args:
            relations: Liste des relations à optimiser (chaque relation doit avoir child_uri et parent_uri).

        Returns:
            Liste des relations hiérarchiques optimisées.
        """
        logger.info("Début de l'optimisation hiérarchique...")
        start_time = time.time()
        self.stats = OptimizationStats(initial_relations_count=len(relations)) # Réinitialiser les stats

        try:
            # 1. Suppression des redondances transitives
            relations = self.remove_redundancies(relations)
            
            # Reconstruire le graphe après suppression des redondances pour la détection de cycles
            graph = self.build_hierarchy_graph(relations)

            # 2. Détection et résolution des cycles
            cycles = self.detect_cycles(graph)
            if cycles:
                relations = self.resolve_cycles(relations, cycles)
                # Reconstruire le graphe après résolution des cycles pour la validation poly-hiérarchie
                graph = self.build_hierarchy_graph(relations)

            # 3. Validation de la poly-hiérarchie
            relations = self.validate_poly_hierarchy(relations)

            self.stats.final_relations_count = len(relations)
            self.stats.execution_time = time.time() - start_time
            
            # Statistiques finales
            logger.info(f"Optimisation terminée en {self.stats.execution_time:.3f}s")
            logger.info(f"Statistiques: {self.stats.initial_relations_count} -> "
                           f"{self.stats.final_relations_count} relations "
                           f"({self.stats.redundancies_removed} redondances, "
                           f"{self.stats.cycles_resolved} cycles résolus, "
                           f"{self.stats.poly_hierarchy_adjustments} ajustements poly-hiérarchie)")
            
            return relations
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation hiérarchique: {e}")
            raise

    def get_optimization_stats(self) -> OptimizationStats:
        """
        Retourne les statistiques de la dernière optimisation.
        
        Returns:
            OptimizationStats: Statistiques détaillées
        """
        return self.stats