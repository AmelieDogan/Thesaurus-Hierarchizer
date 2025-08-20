"""
Module HierarchyBuilder - Phase 5 : Construction de hiérarchie thésaurus SKOS

Ce module construit une hiérarchie SKOS optimisée à partir des relations extraites 
par les phases précédentes, avec validation manuelle des parents candidats et 
résolution intelligente des conflits.

Le processus de construction suit 5 étapes principales :
    - Phase 5A : Préparation et tri des relations
    - Phase 5B : Validation interactive des candidats
    - Phase 5C : Résolution des poly-hiérarchies
    - Phase 5D : Élimination des redondances transitives
    - Phase 5E : Validation finale et export

Classes:
    HierarchyBuilderConfig: Configuration pour le constructeur de hiérarchie
    HierarchyBuilder: Constructeur principal de hiérarchie thésaurus SKOS

Example:
    >>> config = HierarchyBuilderConfig()
    >>> builder = HierarchyBuilder(lexical_indexes, "http://example.org/", config=config)
    >>> hierarchy_graph, enriched_indexes, report = builder.build_hierarchy(relations)
"""

import uuid
from typing import Dict, List, Tuple, Any, Set, Optional
from datetime import datetime
from collections import defaultdict, deque
import json
import logging

from .utils import normalize_text_for_grouping, group_candidate_parents

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchyBuilderConfig:
    """
    Configuration pour le constructeur de hiérarchie.

    Cette classe centralise tous les paramètres configurables pour le processus
    de construction de hiérarchie, permettant une personnalisation fine du comportement.

    Attributes:
        max_parents (int): Nombre maximum de parents par terme (défaut: 3)
        min_confidence_threshold (float): Seuil minimal de confiance pour les relations (défaut: 0.1)
        auto_validate_high_confidence (float): Seuil pour validation automatique (défaut: 0.95)
        batch_size_validation (int): Taille des lots pour validation (défaut: 10)
        enable_transitive_reduction (bool): Active la réduction transitive (défaut: True)
        cycle_detection_enabled (bool): Active la détection de cycles (défaut: True)
        save_intermediate_states (bool): Sauvegarde les états intermédiaires (défaut: True)

    Example:
        >>> config = HierarchyBuilderConfig()
        >>> config.max_parents = 2
        >>> config.min_confidence_threshold = 0.2
    """
    
    def __init__(self):
        self.max_parents: int = 3
        self.min_confidence_threshold: float = 0.1
        self.auto_validate_high_confidence: float = 0.95
        self.batch_size_validation: int = 10
        self.enable_transitive_reduction: bool = True
        self.cycle_detection_enabled: bool = True
        self.save_intermediate_states: bool = True


class HierarchyBuilder:
    """
    Constructeur de hiérarchie thésaurus SKOS.
    
    Cette classe implémente la Phase 5 du processus de construction de thésaurus,
    en transformant les relations extraites en une hiérarchie SKOS validée.
    """
    
    def __init__(self, lexical_indexes: Dict[str, Any], uri_base: str, 
                 max_parents: int = 3, config: Optional[HierarchyBuilderConfig] = None):
        """
        Initialise le constructeur de hiérarchie.

        Args:
            lexical_indexes (Dict[str, Any]): Index lexicaux de la Phase 1 contenant :
                - 'preflabel_to_uri': Mapping prefLabel → URI
                - 'uri_to_preflabel': Mapping URI → prefLabel  
                - 'altlabel_to_uri': Mapping altLabel → liste d'URIs
            uri_base (str): Base URI pour la création de nouveaux termes
            max_parents (int, optional): Nombre maximum de parents par terme. Défaut: 3
            config (HierarchyBuilderConfig, optional): Configuration personnalisée

        Attributes:
            lexical_indexes (Dict[str, Any]): Index lexicaux fournis
            uri_base (str): Base URI pour nouveaux termes
            max_parents (int): Limite du nombre de parents
            config (HierarchyBuilderConfig): Configuration utilisée
            validation_log (List[Dict]): Journal des validations utilisateur
            created_terms (List[Dict]): Termes créés durant la validation
            build_start_time (datetime): Horodatage de début de construction
            user_interactions (int): Compteur d'interactions utilisateur

        Example:
            >>> indexes = {'preflabel_to_uri': {}, 'uri_to_preflabel': {}, 'altlabel_to_uri': {}}
            >>> builder = HierarchyBuilder(indexes, "http://example.org/thesaurus/")
        """
        self.lexical_indexes = lexical_indexes
        self.uri_base = uri_base
        self.max_parents = max_parents
        self.config = config or HierarchyBuilderConfig()
        
        # État interne
        self.validation_log = []
        self.created_terms = []
        self.build_start_time = None
        self.user_interactions = 0
        
    def build_hierarchy(self, all_relations_raw: List[Dict]) -> Tuple[Dict, Dict, Dict]:
        """
        Méthode principale - construit la hiérarchie complète.

        Cette méthode orchestre l'ensemble du processus de construction de hiérarchie
        en 5 phases séquentielles avec validation utilisateur et résolution de conflits.

        Args:
            all_relations_raw (List[Dict]): Relations extraites des phases précédentes.
                Chaque relation doit contenir les champs :
                - 'child': nom du terme enfant
                - 'parent': nom du terme parent
                - 'child_uri': URI du terme enfant
                - 'parent_uri': URI du terme parent
                - 'confidence': score de confiance [0.0-1.0]
                - 'type': 'existing_parent' ou 'candidate_parent'
                - 'source': source de la relation

        Returns:
            Tuple[Dict, Dict, Dict]: Un tuple contenant :
                - hierarchy_graph: Graphe de hiérarchie structuré avec 'nodes' et 'relations'
                - enriched_indexes: Index lexicaux enrichis des nouveaux termes
                - build_report: Rapport détaillé de construction avec métriques

        Raises:
            ValueError: Si les données d'entrée sont invalides ou incohérentes
            Exception: Pour toute erreur durant le processus de construction

        Example:
            >>> relations = [
            ...     {
            ...         'child': 'chat', 'parent': 'animal', 
            ...         'child_uri': 'uri:chat', 'parent_uri': 'uri:animal',
            ...         'confidence': 0.9, 'type': 'existing_parent', 'source': 'phase2'
            ...     }
            ... ]
            >>> graph, indexes, report = builder.build_hierarchy(relations)
        """
        logger.info("Démarrage de la construction de hiérarchie")
        self.build_start_time = datetime.now()
        
        try:
            # Phase 5A : Préparation et tri des relations
            logger.info("Phase 5A : Préparation et tri des relations")
            self._validate_input_data(all_relations_raw, self.lexical_indexes)
            existing_relations, candidate_relations = self._separate_relations_by_type(all_relations_raw)
            grouped_candidates = self._group_candidates_by_parent(candidate_relations)
            hierarchy_graph = self._build_initial_graph(existing_relations)

            logger.info("Début de la validation interactive des parents candidats...")
            candidate_parents_list = list(set([
                rel['parent'] for rel in candidate_relations
            ]))
            
            # Utilisation de la nouvelle fonction utilitaire pour regrouper les parents
            grouped_candidates = group_candidate_parents(candidate_parents_list)
            
            # Phase 5B : Validation interactive des candidats
            logger.info("Phase 5B : Validation interactive des candidats")
            validated_candidates = self._validate_candidate_parents(grouped_candidates, candidate_relations)
            
            # Fusion des relations validées
            all_validated_relations = existing_relations + validated_candidates
            
            # Phase 5C : Résolution des poly-hiérarchies
            logger.info("Phase 5C : Résolution des poly-hiérarchies")
            conflicts = self._identify_multiple_parent_conflicts(all_validated_relations)
            final_relations = self._resolve_parent_conflicts(conflicts, all_validated_relations)
            
            # Mise à jour du graphe avec les relations finales
            hierarchy_graph = self._build_final_graph(final_relations)
            
            # Phase 5D : Élimination des redondances transitives
            if self.config.enable_transitive_reduction:
                logger.info("Phase 5D : Élimination des redondances transitives")
                redundant_relations = self._detect_transitive_redundancies(hierarchy_graph)
                hierarchy_graph = self._remove_redundant_relations(hierarchy_graph, redundant_relations)
            
            # Phase 5E : Validation finale et export
            logger.info("Phase 5E : Validation finale et export")
            if self.config.cycle_detection_enabled:
                cycles = self._detect_cycles(hierarchy_graph)
                if cycles:
                    logger.warning(f"Cycles détectés : {len(cycles)}")
            
            # Enrichissement des index
            enriched_indexes = self._enrich_lexical_indexes()
            
            # Génération du rapport
            build_report = self._generate_build_report(hierarchy_graph, all_relations_raw)
            
            logger.info("Construction de hiérarchie terminée avec succès")
            return hierarchy_graph, enriched_indexes, build_report
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction : {e}")
            raise

    def _get_new_uri(self, term: str) -> str:
        """Génère un nouvel URI pour un terme."""
        return f"{self.uri_base}/vocabulary/{str(uuid.uuid4())}"
    
    def _validate_input_data(self, all_relations_raw: List[Dict], lexical_indexes: Dict) -> None:
        """
        Valide la cohérence des données d'entrée.
        
        Vérifie la présence des champs requis, la validité des scores de confiance
        et l'absence de relations auto-référentielles.
        
        Args:
            all_relations_raw (List[Dict]): Relations à valider
            lexical_indexes (Dict): Index lexicaux de référence
            
        Raises:
            ValueError: Si des champs sont manquants ou des valeurs invalides
        """
        logger.debug("Validation des données d'entrée")
        
        required_fields = ['child', 'parent', 'child_uri', 'parent_uri', 
                          'confidence', 'type', 'source']
        
        for i, relation in enumerate(all_relations_raw):
            # Vérification des champs requis
            for field in required_fields:
                if field not in relation:
                    raise ValueError(f"Champ manquant '{field}' dans la relation {i}")
            
            # Validation du score de confiance
            confidence = relation['confidence']
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Score de confiance invalide : {confidence}")
            
            # Validation auto-référentielle
            if relation['child_uri'] == relation['parent_uri']:
                raise ValueError(f"Relation auto-référentielle détectée : {relation['child_uri']}")
        
        logger.debug(f"Validation réussie pour {len(all_relations_raw)} relations")
    
    def _separate_relations_by_type(self, all_relations_raw: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Sépare les relations existantes des candidats.
        
        Args:
            all_relations_raw (List[Dict]): Toutes les relations d'entrée
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (relations_existantes, relations_candidates)
        """
        existing_relations = []
        candidate_relations = []
        
        for relation in all_relations_raw:
            if relation['type'] == 'existing_parent':
                existing_relations.append(relation)
            elif relation['type'] == 'candidate_parent':
                candidate_relations.append(relation)
        
        logger.info(f"Relations existantes : {len(existing_relations)}, "
                   f"Candidats : {len(candidate_relations)}")
        return existing_relations, candidate_relations
    
    def _group_candidates_by_parent(self, candidate_relations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Groupe les relations candidats par terme parent.
        
        Les groupes sont triés par fréquence décroissante pour optimiser
        l'efficacité de la validation utilisateur.
        
        Args:
            candidate_relations (List[Dict]): Relations candidates à grouper
            
        Returns:
            Dict[str, List[Dict]]: Mapping terme_parent → liste_relations
        """
        grouped = defaultdict(list)
        
        for relation in candidate_relations:
            parent_term = relation['parent']
            grouped[parent_term].append(relation)
        
        # Tri par fréquence décroissante
        sorted_groups = dict(sorted(grouped.items(), 
                                   key=lambda x: len(x[1]), reverse=True))
        
        logger.info(f"Groupement en {len(sorted_groups)} parents candidats")
        return sorted_groups
    
    def _build_initial_graph(self, existing_relations: List[Dict]) -> Dict:
        """
        Construit le graphe initial avec les relations certaines.
        
        Initialise tous les nœuds à partir de l'index lexical existant
        et ajoute les relations déjà validées.
        
        Args:
            existing_relations (List[Dict]): Relations déjà validées
            
        Returns:
            Dict: Graphe avec structure {'nodes': dict, 'relations': list}
        """
        nodes = {}
        relations = []
        
        # Initialisation des nœuds à partir de l'index existant
        for uri, preflabel in self.lexical_indexes['uri_to_preflabel'].items():
            nodes[uri] = {
                'preflabel': preflabel,
                'uri': uri,
                'status': 'existing',
                'children': [],
                'parents': [],
                'metadata': {
                    'creation_source': 'existing_index',
                    'validation_timestamp': datetime.now().isoformat(),
                    'user_validated': False
                }
            }
        
        # Ajout des relations existantes
        for relation in existing_relations:
            child_uri = relation['child_uri']
            parent_uri = relation['parent_uri']
            
            if parent_uri and child_uri in nodes and parent_uri in nodes:
                nodes[child_uri]['parents'].append(parent_uri)
                nodes[parent_uri]['children'].append(child_uri)
                
                relations.append({
                    'child_uri': child_uri,
                    'parent_uri': parent_uri,
                    'confidence': relation['confidence'],
                    'source': relation['source'],
                    'validation_status': 'auto'
                })
        
        return {'nodes': nodes, 'relations': relations}
    
    def _validate_candidate_parents(self, grouped_candidates: Dict[str, List[Dict]], candidate_relations) -> List[Dict]:
        """
        Interface de validation manuelle des parents candidats.
        
        Présente une interface interactive permettant à l'utilisateur
        de valider, rejeter ou modifier chaque parent candidat.
        
        Args:
            grouped_candidates (Dict[str, List[Dict]]): Candidats groupés par parent
            
        Returns:
            List[Dict]: Relations validées par l'utilisateur
        """
        validated_relations = []
        
        print("\n" + "="*80)
        print("VALIDATION DES PARENTS CANDIDATS")
        print("="*80)
        print(f"Nombre de groupes de parents candidats à valider : {len(grouped_candidates)}")
        print("\nOptions disponibles :")
        print("    a - Accepter toutes les relations pour ce groupe")
        print("    r - Rejeter toutes les relations pour ce groupe")
        print("    m - Modifier (proposer un terme alternatif)")
        print("    s - Skip (passer à plus tard)")
        print("    q - Quit (sauvegarder et quitter)")
        print("-"*80)
        
        # Création d'un index rapide pour les relations par terme original
        relations_by_parent_term = defaultdict(list)
        for rel in candidate_relations:
            relations_by_parent_term[rel['parent']].append(rel)
        
        for normalized_term, original_terms in grouped_candidates.items():
            # Affiche les détails du groupe à l'utilisateur
            choice = self._display_candidate_validation_prompt_for_group(
                normalized_term, original_terms, relations_by_parent_term
            )
            
            if choice.upper() == 'q':
                print("Arrêt de la validation demandé par l'utilisateur")
                break
            
            validated_and_updated_relations = self._process_user_choice_for_group(
                choice, normalized_term, original_terms, relations_by_parent_term
            )
            validated_relations.extend(validated_and_updated_relations)
            self.user_interactions += 1
            
            if self.config.save_intermediate_states and len(validated_relations) % 50 == 0:
                logger.info(f"Sauvegarde intermédiaire : {len(validated_relations)} relations validées")
                
        return validated_relations
    
    def _display_candidate_validation_prompt_for_group(
        self, 
        normalized_term: str, 
        original_terms: Set[str],
        relations_by_parent_term: Dict[str, List[Dict]]
    ) -> str:
        """
        Affiche un prompt de validation pour un groupe de termes parents et renvoie le choix de l'utilisateur.
        """
        print(f"\n📂 Groupe de parents : '{normalized_term}'")
        print(f"   Variantes originales : {', '.join(original_terms)}")

        # Collecte tous les enfants des variantes
        all_children = []
        for term in original_terms:
            all_children.extend(relations_by_parent_term.get(term, []))

        print(f"   Nombre total d'enfants proposés : {len(all_children)}")

        # Affiche jusqu'à 10 enfants pour donner du contexte
        print("   Exemples d'enfants proposés :")
        for i, relation in enumerate(all_children[:10], 1):
            child = relation['child']
            confidence = relation.get('confidence', None)
            if confidence is not None:
                print(f"     {i:2d}. {child} (confiance: {confidence:.2f})")
            else:
                print(f"     {i:2d}. {child}")

        if len(all_children) > 10:
            print(f"     ... et {len(all_children) - 10} autres")

        # Statistiques
        if all_children:
            avg_conf = sum(r.get('confidence', 1.0) for r in all_children) / len(all_children)
            print(f"   Confiance moyenne : {avg_conf:.2f}")

        # Boucle de validation du choix
        while True:
            choice = input(
                f"\n🤔 Action pour le groupe '{normalized_term}' [A/R/M/I/S/Q] ? "
            ).strip().upper()
            if choice in ['A', 'R', 'M', 'I', 'S', 'Q']:
                return choice
            print("❌ Choix invalide. Utilisez A, R, M, I, S ou Q")


    def _process_user_choice_for_group(
        self, 
        choice: str, 
        normalized_term: str,
        original_terms: Set[str],
        relations_by_parent_term: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """
        Traite le choix de l'utilisateur pour un groupe de parents et retourne les relations validées.
        """
        validated_relations = []
        choice = choice.upper()
        timestamp = datetime.now().isoformat()

        # Rassemble toutes les relations du groupe
        group_relations = []
        for term in original_terms:
            group_relations.extend(relations_by_parent_term.get(term, []))

        if choice == 'A':  # Accepter
            new_uri = self._get_new_uri(normalized_term)
            for rel in group_relations:
                rel['parent_uri'] = new_uri
                rel['parent'] = normalized_term
                validated_relations.append(rel)
            print(f"✅ Groupe '{normalized_term}' accepté ({len(validated_relations)} relations)")
            self.validation_log.append({
                'candidate_parent': normalized_term,
                'action_taken': 'accepted',
                'alternative_term': None,
                'child_relations_count': len(validated_relations),
                'timestamp': timestamp
            })

        elif choice == 'R':  # Rejeter
            print(f"❌ Groupe '{normalized_term}' rejeté ({len(group_relations)} relations ignorées)")
            self.validation_log.append({
                'candidate_parent': normalized_term,
                'action_taken': 'rejected',
                'alternative_term': None,
                'child_relations_count': len(group_relations),
                'timestamp': timestamp
            })

        elif choice == 'M':  # Modifier
            new_term = input(f"🔄 Nouveau terme pour le groupe '{normalized_term}' : ").strip()
            if new_term:
                new_uri = self._get_new_uri(new_term)
                for rel in group_relations:
                    rel['parent_uri'] = new_uri
                    rel['parent'] = new_term
                    validated_relations.append(rel)
                print(f"✏️ Groupe '{normalized_term}' renommé en '{new_term}' ({len(validated_relations)} relations)")
                self.validation_log.append({
                    'candidate_parent': normalized_term,
                    'action_taken': 'modified',
                    'alternative_term': new_term,
                    'child_relations_count': len(validated_relations),
                    'timestamp': timestamp
                })

        elif choice == 'I':  # Validation individuelle
            validated_relations = self._validate_individual_relations(normalized_term, group_relations)

        elif choice == 'S':  # Skip
            print(f"⏭️  Groupe '{normalized_term}' passé (à traiter plus tard)")
            self.validation_log.append({
                'candidate_parent': normalized_term,
                'action_taken': 'skipped',
                'alternative_term': None,
                'child_relations_count': 0,
                'timestamp': timestamp
            })

        elif choice == 'Q':
            pass  # Gestion du quit en amont probablement

        return validated_relations

    
    def _handle_alternative_term(self, alternative_term: str, original_relations: List[Dict]) -> List[Dict]:
        """
        Gère la proposition d'un terme alternatif par l'utilisateur.
        
        Recherche le terme dans les index existants ou crée un nouveau terme,
        puis met à jour toutes les relations concernées.
        
        Args:
            alternative_term (str): Terme alternatif proposé
            original_relations (List[Dict]): Relations originales à modifier
            
        Returns:
            List[Dict]: Relations mises à jour avec le nouveau parent
        """
        # Normalisation du terme alternatif
        alt_normalized = alternative_term.lower().strip()
        
        # Recherche dans l'index existant
        alternative_uri = None
        
        # Recherche par prefLabel
        if alt_normalized in self.lexical_indexes['preflabel_to_uri']:
            alternative_uri = self.lexical_indexes['preflabel_to_uri'][alt_normalized]
            print(f"🔍 Terme trouvé dans l'index existant : {alternative_uri}")
            
        # Recherche par altLabel
        elif alt_normalized in self.lexical_indexes['altlabel_to_uri']:
            uris = self.lexical_indexes['altlabel_to_uri'][alt_normalized]
            if len(uris) == 1:
                alternative_uri = uris[0]
                print(f"🔍 Terme trouvé comme altLabel : {alternative_uri}")
            else:
                print(f"⚠️  Terme ambigu ({len(uris)} correspondances)")
                return []
        
        # Création d'un nouveau terme si non trouvé
        if not alternative_uri:
            alternative_uri = f"{self.uri_base}/{uuid.uuid4()}"
            print(f"🆕 Nouveau terme créé : {alternative_uri}")
            
            # Ajout aux index enrichis
            self.lexical_indexes['preflabel_to_uri'][alt_normalized] = alternative_uri
            self.lexical_indexes['uri_to_preflabel'][alternative_uri] = alternative_term
            
            self.created_terms.append({
                'uri': alternative_uri,
                'preflabel': alternative_term,
                'created_from': 'user_alternative',
                'timestamp': datetime.now().isoformat()
            })
        
        # Mise à jour des relations avec le nouveau parent
        modified_relations = []
        for relation in original_relations:
            new_relation = relation.copy()
            new_relation['parent'] = alternative_term
            new_relation['parent_uri'] = alternative_uri
            new_relation['type'] = 'existing_parent'  # Devient une relation validée
            modified_relations.append(new_relation)
        
        print(f"✅ {len(modified_relations)} relations mises à jour avec '{alternative_term}'")
        return modified_relations
    
    def _validate_individual_relations(self, parent_term: str, relations: List[Dict]) -> List[Dict]:
        """
        Interface pour valider chaque relation individuellement.
        
        Args:
            parent_term (str): Terme parent concerné
            relations (List[Dict]): Relations à valider individuellement
            
        Returns:
            List[Dict]: Relations validées
        """
        validated = []
        print(f"\n🔍 Validation individuelle pour '{parent_term}':")
        
        for i, relation in enumerate(relations, 1):
            child = relation['child']
            confidence = relation['confidence']
            
            while True:
                choice = input(f"  {i}/{len(relations)} - '{child}' -> '{parent_term}' "
                              f"(confiance: {confidence:.2f}) [O/N/Q] ? ").strip().upper()
                
                if choice == 'O':
                    validated.append(relation)
                    print(f"    ✅ Relation validée")
                    break
                elif choice == 'N':
                    print(f"    ❌ Relation rejetée")
                    break
                elif choice == 'Q':
                    print(f"    🛑 Arrêt de la validation individuelle")
                    return validated
                else:
                    print("    ❌ Choix invalide. Utilisez O (Oui), N (Non) ou Q (Quitter)")
        
        print(f"✅ Validation individuelle terminée : {len(validated)}/{len(relations)} relations validées")
        return validated
    
    def _identify_multiple_parent_conflicts(self, all_validated_relations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Identifie les termes avec plus de max_parents parents potentiels.
        
        Args:
            all_validated_relations (List[Dict]): Toutes les relations validées
            
        Returns:
            Dict[str, List[Dict]]: Mapping URI_enfant → liste_relations_conflictuelles
        """
        child_to_relations = defaultdict(list)
        
        # Groupement par enfant
        for relation in all_validated_relations:
            child_uri = relation['child_uri']
            child_to_relations[child_uri].append(relation)
        
        # Identification des conflits
        conflicts = {}
        for child_uri, child_relations in child_to_relations.items():
            if len(child_relations) > self.max_parents:
                conflicts[child_uri] = child_relations
        
        if conflicts:
            logger.info(f"Conflits de poly-hiérarchie détectés : {len(conflicts)} termes")
        
        return conflicts
    
    def _resolve_parent_conflicts(self, conflicts: Dict[str, List[Dict]], 
                                 all_relations: List[Dict]) -> List[Dict]:
        """
        Résout les conflits en appliquant la règle des scores égaux.
        
        Pour chaque terme en conflit, sélectionne les parents ayant les scores
        les plus élevés dans la limite de max_parents.
        
        Args:
            conflicts (Dict[str, List[Dict]]): Conflits identifiés
            all_relations (List[Dict]): Toutes les relations
            
        Returns:
            List[Dict]: Relations finales après résolution
        """
        final_relations = []
        
        # Relations sans conflit
        conflict_uris = set(conflicts.keys())
        for relation in all_relations:
            if relation['child_uri'] not in conflict_uris:
                final_relations.append(relation)
        
        # Résolution des conflits
        for child_uri, child_relations in conflicts.items():
            resolved = self._apply_equal_score_rule(child_relations)
            final_relations.extend(resolved)
            
            child_term = child_relations[0]['child']
            logger.info(f"Conflit résolu pour '{child_term}' : "
                       f"{len(child_relations)} -> {len(resolved)} parents")
        
        return final_relations
    
    def _apply_equal_score_rule(self, relations: List[Dict]) -> List[Dict]:
        """
        Applique la règle des scores égaux pour la résolution de conflits.
        
        Sélectionne les relations par groupes de scores décroissants,
        en prenant tous les éléments d'un même groupe si possible.
        
        Args:
            relations (List[Dict]): Relations en conflit pour un terme
            
        Returns:
            List[Dict]: Relations sélectionnées selon la règle
        """
        # Groupement par score de confiance
        score_groups = self._group_relations_by_confidence(relations)
        
        # Tri des scores par ordre décroissant
        sorted_scores = sorted(score_groups.keys(), reverse=True)
        
        selected_relations = []
        for score in sorted_scores:
            group_relations = score_groups[score]
            
            # Si ajouter tout le groupe ne dépasse pas la limite
            if len(selected_relations) + len(group_relations) <= self.max_parents:
                selected_relations.extend(group_relations)
            # Si on n'a pas encore atteint la limite mais que le groupe la fait dépasser
            elif len(selected_relations) < self.max_parents:
                selected_relations.extend(group_relations)  # On prend tout le groupe
                break
            else:
                break  # On a déjà assez de relations
        
        return selected_relations
    
    def _group_relations_by_confidence(self, relations: List[Dict]) -> Dict[float, List[Dict]]:
        """
        Groupe les relations par score de confiance.
        
        Args:
            relations (List[Dict]): Relations à grouper
            
        Returns:
            Dict[float, List[Dict]]: Mapping score → liste_relations
        """
        grouped = defaultdict(list)
        
        for relation in relations:
            confidence = relation['confidence']
            grouped[confidence].append(relation)
        
        return dict(grouped)
    
    def _build_final_graph(self, final_relations: List[Dict]) -> Dict:
        """Construit le graphe final avec toutes les relations validées."""
        # Réinitialisation des listes de parents/enfants
        hierarchy_graph = self._build_initial_graph([])
        
        # Ajout de tous les nouveaux termes créés
        for term_info in self.created_terms:
            uri = term_info['uri']
            preflabel = term_info['preflabel']
            
            hierarchy_graph['nodes'][uri] = {
                'preflabel': preflabel,
                'uri': uri,
                'status': 'user_created',
                'children': [],
                'parents': [],
                'metadata': {
                    'creation_source': term_info['created_from'],
                    'validation_timestamp': term_info['timestamp'],
                    'user_validated': True
                }
            }
        
        # Construction des relations
        hierarchy_graph['relations'] = []
        
        for relation in final_relations:
            child_uri = relation['child_uri']
            parent_uri = relation['parent_uri']
            
            if parent_uri and child_uri in hierarchy_graph['nodes'] and parent_uri in hierarchy_graph['nodes']:
                # Mise à jour des listes de liens
                if parent_uri not in hierarchy_graph['nodes'][child_uri]['parents']:
                    hierarchy_graph['nodes'][child_uri]['parents'].append(parent_uri)
                
                if child_uri not in hierarchy_graph['nodes'][parent_uri]['children']:
                    hierarchy_graph['nodes'][parent_uri]['children'].append(child_uri)
                
                # Ajout à la liste des relations
                validation_status = 'user_validated' if relation.get('type') == 'existing_parent' else 'auto'
                
                hierarchy_graph['relations'].append({
                    'child_uri': child_uri,
                    'parent_uri': parent_uri,
                    'confidence': relation['confidence'],
                    'source': relation['source'],
                    'validation_status': validation_status
                })
        
        return hierarchy_graph
    
    def _detect_transitive_redundancies(self, hierarchy_graph: Dict) -> List[Tuple[str, str]]:
        """
        Détecte les relations A→C redondantes quand A→B→C existe.
        
        Identifie les relations directes qui peuvent être déduites
        par transitivité d'autres chemins dans la hiérarchie.
        
        Args:
            hierarchy_graph (Dict): Graphe de hiérarchie
            
        Returns:
            List[Tuple[str, str]]: Liste des relations redondantes (parent_uri, child_uri)
        """
        redundant_relations = []
        
        # Construction d'un graphe d'adjacence pour la recherche de chemins
        adjacency = defaultdict(list)
        direct_relations = set()
        
        for relation in hierarchy_graph['relations']:
            parent_uri = relation['parent_uri']
            child_uri = relation['child_uri']
            adjacency[parent_uri].append(child_uri)
            direct_relations.add((parent_uri, child_uri))
        
        # Vérification de chaque relation directe
        for parent_uri, child_uri in direct_relations:
            # Recherche de chemins alternatifs (longueur > 1)
            alternative_paths = self._find_paths_excluding_direct(
                adjacency, parent_uri, child_uri, max_depth=5
            )
            
            if alternative_paths:
                redundant_relations.append((parent_uri, child_uri))
                logger.debug(f"Relation redondante détectée : {parent_uri} -> {child_uri}")
        
        logger.info(f"Relations redondantes détectées : {len(redundant_relations)}")
        return redundant_relations
    
    def _find_paths_excluding_direct(self, adjacency: Dict, source: str, 
                                   target: str, max_depth: int = 5) -> List[List[str]]:
        """
        Trouve tous les chemins entre source et target excluant le lien direct.
        
        Args:
            adjacency (Dict): Graphe d'adjacence
            source (str): URI source
            target (str): URI cible
            max_depth (int): Profondeur maximale de recherche
            
        Returns:
            List[List[str]]: Chemins alternatifs trouvés
        """
        paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
                
            if current == target and len(path) > 2:  # Chemin avec au moins un intermédiaire
                paths.append(path.copy())
                return
            
            if current in adjacency:
                for neighbor in adjacency[current]:
                    # Éviter les cycles et le lien direct source→target
                    if neighbor not in path and not (current == source and neighbor == target):
                        path.append(neighbor)
                        dfs(neighbor, path, depth + 1)
                        path.pop()
        
        dfs(source, [source], 0)
        return paths
    
    def _remove_redundant_relations(self, hierarchy_graph: Dict, 
                                   redundant_relations: List[Tuple]) -> Dict:
        """
        Supprime les relations redondantes du graphe.
        
        Args:
            hierarchy_graph (Dict): Graphe original
            redundant_relations (List[Tuple]): Relations à supprimer
            
        Returns:
            Dict: Graphe nettoyé
        """
        redundant_set = set(redundant_relations)
        
        # Filtrage des relations
        filtered_relations = []
        for relation in hierarchy_graph['relations']:
            parent_uri = relation['parent_uri']
            child_uri = relation['child_uri']
            
            if (parent_uri, child_uri) not in redundant_set:
                filtered_relations.append(relation)
        
        # Mise à jour des listes parents/enfants dans les nœuds
        for node in hierarchy_graph['nodes'].values():
            node['parents'] = []
            node['children'] = []
        
        for relation in filtered_relations:
            parent_uri = relation['parent_uri']
            child_uri = relation['child_uri']
            
            hierarchy_graph['nodes'][child_uri]['parents'].append(parent_uri)
            hierarchy_graph['nodes'][parent_uri]['children'].append(child_uri)
        
        hierarchy_graph['relations'] = filtered_relations
        
        logger.info(f"Relations redondantes supprimées : {len(redundant_relations)}")
        return hierarchy_graph
    
    def _detect_cycles(self, hierarchy_graph: Dict) -> List[List[str]]:
        """
        Détecte les cycles dans la hiérarchie.

        Utilise un parcours en profondeur (DFS) avec marquage des états
        pour identifier tous les cycles présents.

        Args:
            hierarchy_graph (Dict): Graphe de hiérarchie

        Returns:
            List[List[str]]: Liste des cycles détectés, chaque cycle étant une liste d'URIs
        """
        cycles = []
        # États des nœuds : 0=non visité, 1=en cours de visite, 2=visité
        visited_states = {uri: 0 for uri in hierarchy_graph['nodes']}
        
        def find_cycles_dfs(node: str, path: List[str]):
            visited_states[node] = 1  # Marque le nœud comme en cours de visite
            path.append(node)
            
            for neighbor in hierarchy_graph['nodes'][node]['children']:
                if neighbor not in visited_states:
                    # Gère le cas où un enfant n'est pas dans le graphe (données incohérentes)
                    logger.warning(f"Le nœud voisin '{neighbor}' n'est pas dans le graphe. Ignoré.")
                    continue

                if visited_states[neighbor] == 1:
                    # Cycle détecté
                    try:
                        cycle_start_index = path.index(neighbor)
                        cycles.append(path[cycle_start_index:])
                    except ValueError:
                        logger.warning(
                            f"Nœud de début de cycle '{neighbor}' non trouvé dans le chemin. "
                            "Cela indique une incohérence de l'état."
                        )
                elif visited_states[neighbor] == 0:
                    find_cycles_dfs(neighbor, path)
                    
            visited_states[node] = 2  # Marque le nœud comme visité
            path.pop()

        for node_uri in hierarchy_graph['nodes']:
            if visited_states[node_uri] == 0:
                find_cycles_dfs(node_uri, [])
                
        if cycles:
            logger.warning(f"Cycles détectés dans la hiérarchie : {len(cycles)}")
            
        return cycles
    
    def _calculate_hierarchy_metrics(self, hierarchy_graph: Dict) -> Dict[str, Any]:
        """
        Calcule les métriques de qualité de la hiérarchie.
        
        Args:
            hierarchy_graph (Dict): Graphe de hiérarchie
            
        Returns:
            Dict[str, Any]: Métriques incluant :
                - total_nodes: nombre de nœuds
                - total_relations: nombre de relations
                - max_depth: profondeur maximale
                - average_depth: profondeur moyenne
                - root_terms_count: nombre de termes racines
                - leaf_terms_count: nombre de termes feuilles
                - avg_parents_per_term: moyenne de parents par terme
                - avg_children_per_term: moyenne d'enfants par terme
        """
        nodes = hierarchy_graph['nodes']
        relations = hierarchy_graph['relations']
        
        # Métriques de base
        total_nodes = len(nodes)
        total_relations = len(relations)
        
        # Termes racines (sans parents)
        root_terms = [uri for uri, node in nodes.items() if not node['parents']]
        
        # Termes feuilles (sans enfants)
        leaf_terms = [uri for uri, node in nodes.items() if not node['children']]
        
        # Distribution du nombre de parents et enfants
        parent_counts = [len(node['parents']) for node in nodes.values()]
        child_counts = [len(node['children']) for node in nodes.values()]
        
        # Profondeur de la hiérarchie
        depths = self._calculate_depths(hierarchy_graph)
        max_depth = max(depths.values()) if depths else 0
        avg_depth = sum(depths.values()) / len(depths) if depths else 0
        
        # Relations validées manuellement
        user_validated = sum(1 for r in relations 
                           if r['validation_status'] == 'user_validated')
        
        return {
            'total_nodes': total_nodes,
            'total_relations': total_relations,
            'max_depth': max_depth,
            'average_depth': round(avg_depth, 2),
            'root_terms_count': len(root_terms),
            'leaf_terms_count': len(leaf_terms),
            'avg_parents_per_term': round(sum(parent_counts) / len(parent_counts), 2) if parent_counts else 0,
            'avg_children_per_term': round(sum(child_counts) / len(child_counts), 2) if child_counts else 0,
            'user_validated_relations': user_validated,
            'validation_percentage': round(user_validated / total_relations * 100, 2) if total_relations > 0 else 0
        }
    
    def _calculate_depths(self, hierarchy_graph: Dict) -> Dict[str, int]:
        """
        Calcule la profondeur de chaque nœud dans la hiérarchie.
        
        Utilise un parcours en largeur depuis les termes racines
        pour déterminer la profondeur minimale de chaque nœud.
        
        Args:
            hierarchy_graph (Dict): Graphe de hiérarchie
            
        Returns:
            Dict[str, int]: Mapping URI → profondeur
        """
        nodes = hierarchy_graph['nodes']
        depths = {}
        
        # Initialisation avec les termes racines
        root_terms = [uri for uri, node in nodes.items() if not node['parents']]
        queue = deque([(uri, 0) for uri in root_terms])
        
        while queue:
            current_uri, depth = queue.popleft()
            
            if current_uri not in depths or depths[current_uri] > depth:
                depths[current_uri] = depth
                
                # Ajout des enfants avec profondeur + 1
                for child_uri in nodes[current_uri]['children']:
                    queue.append((child_uri, depth + 1))
        
        return depths
    
    def _enrich_lexical_indexes(self) -> Dict[str, Any]:
        """Enrichit les index lexicaux avec les nouveaux termes créés."""
        enriched_indexes = {
            'preflabel_to_uri': self.lexical_indexes['preflabel_to_uri'].copy(),
            'uri_to_preflabel': self.lexical_indexes['uri_to_preflabel'].copy(),
            'altlabel_to_uri': self.lexical_indexes['altlabel_to_uri'].copy()
        }
        
        # Ajout des termes créés pendant la validation
        for term_info in self.created_terms:
            uri = term_info['uri']
            preflabel = term_info['preflabel']
            normalized = preflabel.lower().strip()
            
            enriched_indexes['preflabel_to_uri'][normalized] = uri
            enriched_indexes['uri_to_preflabel'][uri] = preflabel
        
        logger.info(f"Index enrichis avec {len(self.created_terms)} nouveaux termes")
        return enriched_indexes
    
    def _generate_build_report(self, hierarchy_graph: Dict, 
                              original_relations: List[Dict]) -> Dict[str, Any]:
        """
        Génère le rapport de construction.
        
        Args:
            hierarchy_graph (Dict): Graphe final construit
            original_relations (List[Dict]): Relations d'entrée originales
            
        Returns:
            Dict[str, Any]: Rapport complet incluant :
                - summary: résumé statistique
                - hierarchy_metrics: métriques de la hiérarchie
                - validation_details: journal des validations
                - quality_indicators: indicateurs de qualité
                - created_terms: nouveaux termes créés
                - configuration_used: configuration appliquée
        """
        build_duration = (datetime.now() - self.build_start_time).total_seconds()
        hierarchy_metrics = self._calculate_hierarchy_metrics(hierarchy_graph)
        
        # Comptage des actions de validation
        validated_count = sum(1 for log in self.validation_log if log['action_taken'] == 'accepted')
        rejected_count = sum(1 for log in self.validation_log if log['action_taken'] == 'rejected')
        modified_count = sum(1 for log in self.validation_log if log['action_taken'] == 'modified')
        
        # Calcul des indicateurs de qualité
        total_possible_relations = len(original_relations)
        final_relations_count = len(hierarchy_graph['relations'])
        
        completeness_score = (final_relations_count / total_possible_relations * 100) if total_possible_relations > 0 else 0
        
        # Score de consistance (absence de cycles)
        cycles = self._detect_cycles(hierarchy_graph) if self.config.cycle_detection_enabled else []
        consistency_score = max(0, 100 - len(cycles) * 10)  # -10 points par cycle
        
        # Score de couverture (% de termes avec au moins 1 parent)
        terms_with_parents = sum(1 for node in hierarchy_graph['nodes'].values() if node['parents'])
        total_terms = len(hierarchy_graph['nodes'])
        coverage_score = (terms_with_parents / total_terms * 100) if total_terms > 0 else 0
        
        return {
            'summary': {
                'total_input_relations': len(original_relations),
                'validated_relations': validated_count,
                'rejected_relations': rejected_count,
                'modified_relations': modified_count,
                'created_new_terms': len(self.created_terms),
                'build_duration_seconds': round(build_duration, 2),
                'user_interactions_count': self.user_interactions
            },
            'hierarchy_metrics': hierarchy_metrics,
            'validation_details': self.validation_log,
            'quality_indicators': {
                'completeness_score': round(completeness_score, 2),
                'consistency_score': round(consistency_score, 2),
                'coverage_score': round(coverage_score, 2)
            },
            'created_terms': self.created_terms,
            'build_timestamp': datetime.now().isoformat(),
            'configuration_used': {
                'max_parents': self.max_parents,
                'min_confidence_threshold': self.config.min_confidence_threshold,
                'transitive_reduction_enabled': self.config.enable_transitive_reduction,
                'cycle_detection_enabled': self.config.cycle_detection_enabled
            }
        }