import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import Counter
import json
import os

from .config import load_default_config

from .logger import get_logger

logger = get_logger(__name__)

from .phase1_data_processor import ThesaurusDataProcessor
from .phase2_pattern_detector import LexicalPatternDetector
from .phase3_similarity_analyzer import LexicalSimilarityAnalyzer
from .phase4_contextual_embedding_analyzer import ContextualSemanticDiscoveryEngine
from .phase5_hierarchy_builder import HierarchyBuilder
from .phase6_hierarchy_optimizer import HierarchyOptimizer
from .phase7_output_generator import OutputGenerator

class ThesaurusHierarchyBuilder:
    def __init__(self,
                 tsv_file_path: str,
                 config: Dict[str, Any] = None):
        """
        Initialise l’orchestrateur principal.

        Args:
            tsv_file_path: Chemin vers le fichier TSV du thésaurus
            config: Configuration personnalisée (facultatif)
        """
        self.tsv_file_path = tsv_file_path
        self.config = config or load_default_config() # Load default config if none provided

        logger.info("Initialisation de l'orchestrateur principal")
        
        # Initialisation des composants avec les paramètres de configuration
        self.data_processor = ThesaurusDataProcessor(
            tsv_file_path=self.tsv_file_path,
            uri_base=self.config["uri_base"]
        )
        self.lexical_pattern_detector = LexicalPatternDetector(
            min_frequency_for_candidate=self.config["min_frequency_for_candidate"]
        )
        self.lexical_similarity_analyzer = LexicalSimilarityAnalyzer(
            min_substring_length=self.config["min_substring_length"],
            jaccard_threshold=self.config["jaccard_threshold"],
            min_family_size=self.config["min_family_size"],
            max_edit_distance=self.config["max_edit_distance"]
        )
        self.semantic_embedding_analyzer = ContextualSemanticDiscoveryEngine(
            embedding_model_path=self.config["embedding_model_path"],
            context_window_size=self.config["context_window_size"],
            pattern_min_frequency=self.config["pattern_min_frequency"],
            graph_walk_length=self.config["graph_walk_length"],
            zone_coherence_threshold=self.config["zone_coherence_threshold"],
            analogy_similarity_threshold=self.config["analogy_similarity_threshold"],
            abstract_similarity_threshold=self.config["abstract_similarity_threshold"],
            family_similarity_threshold=self.config["family_similarity_threshold"]
        )

        logger.info("Composants de l'orchestrateur initialisés.")

    def run_phase1_data_preparation(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Exécute la Phase 1 : préparation des données.
        
        Returns:
            Tuple: (DataFrame des données préparées, Index lexicaux)
        """
        logger.info("Phase 1 : Préparation des données")
        df, lexical_indexes = self.data_processor.preprocess_data()
        self.stats['phase1'] = self.data_processor.get_statistics()
        return df, lexical_indexes

    def run_phase2_pattern_detection(self, 
                                     df: pd.DataFrame, 
                                     lexical_indexes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Exécute la Phase 2 : détection des relations par patterns lexicaux.
        
        Args:
            df: DataFrame des données préparées.
            lexical_indexes: Index lexicaux.
            
        Returns:
            Liste des relations détectées par patterns.
        """
        logger.info("Phase 2 : Détection des relations par patterns lexicaux")
        pattern_relations = self.lexical_pattern_detector.detect_all_patterns(df, lexical_indexes)
        self.stats['phase2'] = self.lexical_pattern_detector.get_statistics()
        return pattern_relations

    def run_phase3_lexical_similarity_analysis(self, 
                                               df: pd.DataFrame,
                                               lexical_indexes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Exécute la Phase 3 : analyse de similarité lexicale.
        
        Args:
            df: DataFrame des données préparées.
            lexical_indexes: Index lexicaux.
            
        Returns:
            Liste des relations détectées par similarité lexicale.
        """
        logger.info("Phase 3 : Analyse de similarité lexicale")
        lexical_relations = self.lexical_similarity_analyzer.detect_lexical_similarities(df, lexical_indexes)
        self.stats['phase3'] = self.lexical_similarity_analyzer.get_statistics()
        return lexical_relations

    def run_phase4_embedding_analysis(self, 
                                       df: pd.DataFrame, 
                                       all_prev_relations: List[Dict[str, Any]],
                                       lexical_indexes: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """
        Exécute la Phase 4 : analyse sémantique par embeddings.
        
        Args:
            df: DataFrame des données préparées.
            all_prev_relations: Relations détectées par patterns.
            lexical_indexes: Index lexicaux.
            
        Returns:
            Tuple: (Liste des relations sémantiques, Liste des clusters sémantiques)
        """
        logger.info("Phase 4 : Analyse sémantique par embeddings")
        deduplicated_relations, semantic_zones, contextes_enrichis = self.semantic_embedding_analyzer.analyze_embeddings(df, all_prev_relations, lexical_indexes)
        self.stats['phase4'] = self.semantic_embedding_analyzer.get_statistics()
        return deduplicated_relations, semantic_zones, contextes_enrichis

    def run_pipeline(self):
        """
        Exécute le pipeline complet de hiérarchisation du thésaurus.
        """
        self.stats = {} # Reset stats for each run
        
        logger.info("Début du pipeline de hiérarchisation du thésaurus.")
        
        # Phase 1
        df, lexical_indexes, existing_broader_relations = self.data_processor.preprocess_data()

        # Phase 2
        pattern_relations = self.run_phase2_pattern_detection(df, lexical_indexes)

        # Phase 3
        lexical_relations = self.run_phase3_lexical_similarity_analysis(df, lexical_indexes)

        all_prev_relations = existing_broader_relations + pattern_relations + lexical_relations
        
        # Phase 4
        deduplicated_relations, semantic_zones, contextes_enrichis = self.run_phase4_embedding_analysis(df, all_prev_relations, lexical_indexes)

        # Phase 5
        # ======================================================================
        # Intégration et Filtrage des Relations (Orchestrateur)
        # ======================================================================
        logger.info("Phase d'intégration : Fusion et filtrage des relations")

        # Étape de consolidation de toutes les relations brutes avant optimisation
        logger.info("=== Étape de Consolidation: Assemblage de toutes les relations brutes ===")
        all_relations_raw = existing_broader_relations + pattern_relations + lexical_relations + deduplicated_relations
        logger.info(f"Total des relations brutes consolidées avant optimisation : {len(all_relations_raw)}")

        # Ajout de la phase de nettoyage
        logger.info("=== Étape de Nettoyage: Suppression des relations auto-référentielles ===")
        all_relations_raw = self._clean_relations(all_relations_raw)
        logger.info(f"Total des relations après nettoyage : {len(all_relations_raw)}")

        cleaned_relations_raw = self.clean_numpy_types(all_relations_raw)
    
        # Étape de filtrage et combinaison des scores
        final_relations = self._combine_and_filter_relations(cleaned_relations_raw)
        logger.info(f"Relations après combinaison et filtrage initial: {len(final_relations)}")

        builder = HierarchyBuilder(
            lexical_indexes=lexical_indexes,
            uri_base=self.config["uri_base"]
        )
        
        hierarchy, enriched_indexes, report = builder.build_hierarchy(all_relations_raw)

        print("Construction terminée !")
        print(f"Rapport : {json.dumps(report, indent=2, ensure_ascii=False)}")

        # Enregistrement des résultats de la Phase 5
        self.save_to_json(data=hierarchy, filename="final_hierarchy.json")
        self.save_to_json(data=enriched_indexes, filename="enriched_indexes.json")
        self.save_to_json(data=all_relations_raw, filename="all_relations_raw.json")

        # Phase 6 : Optimisation de la hiérarchie
        # A implémenter

        # logger.info(f"Relations finales optimisées: {len(optimized_relations)}")

        # Phase 7 : Génération des sorties
        # A implémenter

        logger.info("Pipeline terminé.")
        logger.info("Statistiques globales du pipeline :")
        for phase, stats_data in self.stats.items():
            logger.info(f"  --- {phase.upper()} ---")
            for key, value in stats_data.items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.3f}")
                else:
                    logger.info(f"    {key}: {value}")

    def save_to_json(self, data, filename: str) -> None:
        """
        Génère un fichier JSON à partir des données fournies, en gérant les types numpy.

        Args:
            data: Les données à sérialiser.
            filename: Le nom du fichier de sortie.
        """
        output_file_path = os.path.join(self.config["output_dir"], filename)

        def clean_value(value):
            if isinstance(value, float) and pd.isna(value):
                return None
            elif isinstance(value, (np.float32, np.float64)):
                return float(value)
            elif isinstance(value, (np.int32, np.int64)):
                return int(value)
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(v) for v in value]
            else:
                return value

        cleaned_data = clean_value(data)

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Données sauvegardées dans : {output_file_path}")
        except TypeError as e:
            logger.error(f"Erreur de sérialisation JSON lors de la sauvegarde de {filename}: {e}")
            logger.error(f"Vérifiez les types de données dans les données de '{filename}'.")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la sauvegarde de {filename}: {e}")

    def _clean_relations(self, relations: List[Dict]) -> List[Dict]:
        """
        Supprime les relations auto-référentielles où l'enfant est son propre parent.
        """
        cleaned_relations = []
        for relation in relations:
            if relation['child_uri'] == relation['parent_uri']:
                logger.warning(
                    f"Relation auto-référentielle détectée et ignorée : "
                    f"{relation['child_uri']} -> {relation['parent_uri']}"
                )
            else:
                cleaned_relations.append(relation)
        return cleaned_relations
    
    def generate_raw_relations_output(self, all_relations_raw: List[Dict[str, Any]]) -> None:
        """
        Génère un fichier JSON contenant toutes les relations brutes avant optimisation.

        Args:
            all_relations_raw: Liste des dictionnaires de relations brutes.
        """
        output_file_path = os.path.join(self.config["output_dir"], "all_relations_raw.json")
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_relations_raw, f, ensure_ascii=False, indent=4)
            logger.info(f"Relations brutes sauvegardées dans : {output_file_path}")
        except TypeError as e:
            logger.error(f"Erreur de sérialisation JSON lors de la sauvegarde des relations brutes: {e}")
            logger.error("Vérifiez les types de données dans 'all_relations_raw'. "
                         "Des valeurs comme 'NaN' ou des types non standards peuvent causer cela.")
            # Optionnel: tenter d'identifier l'élément problématique
            for i, rel in enumerate(all_relations_raw):
                try:
                    json.dumps(rel)
                except TypeError as sub_e:
                    logger.error(f"Erreur de sérialisation dans la relation à l'index {i}: {rel}. Cause: {sub_e}")
                    break # Arrêter après le premier problème trouvé
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la sauvegarde des relations brutes: {e}")

    def clean_numpy_types(self, relations):
        """Nettoie une liste de dictionnaires contenant des types NumPy ou NaN pour une compatibilité JSON."""

        output_file_path = os.path.join(self.config["output_dir"], "all_relations_raw.json")

        def clean_value(value):
            if isinstance(value, float) and pd.isna(value):
                return None
            elif isinstance(value, (np.float32, np.float64)):
                return float(value)
            elif isinstance(value, (np.int32, np.int64)):
                return int(value)
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            else:
                return value

        cleaned_relations = []
        for relation in relations:
            cleaned_relation = {k: clean_value(v) for k, v in relation.items()}
            cleaned_relations.append(cleaned_relation)

        # Sauvegarde pour inspection
        self.generate_raw_relations_output(cleaned_relations)
        logger.info(f"Fichier 'all_relations_raw.json' généré dans le dossier '{output_file_path}' pour inspection.")
        
        return cleaned_relations

    def _combine_and_filter_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine les relations de différentes sources, applique les scores et filtres.
        
        Args:
            relations: Liste de toutes les relations brutes.
            
        Returns:
            Liste des relations finales filtrées et scorées.
        """
        combined_relations = {} # Clé: (child, parent), Valeur: relation avec score combiné
        
        for rel in relations:
            child_parent_key = (rel['child'], rel['parent'])
            
            current_score = rel.get('confidence', 0.0) # Utiliser 'confidence' comme base
            
            # Appliquer des scores spécifiques en fonction du type de relation et de la source
            if rel['source'] == 'existing_skos_broader' or rel['source'] == 'broader_from_preflabel' or rel['source'] == 'pattern_inclusion':
                current_score *= self.config["pattern_exact_score"]
            elif rel['source'] == ('pattern_specialized'):
                current_score *= self.config["pattern_only_score"]
            elif rel['source'] == 'lexical_similarity':
                current_score *= self.config["pattern_only_score"] # Peut être ajusté pour la similarité lexicale
            elif rel['source'] == 'embedding': # Relations sémantiques pures
                current_score *= self.config["embedding_only_score"]
            elif rel['source'] == 'generation':
                # Les candidats ont leur propre confiance, on peut les pondérer
                current_score *= 0.7 # Un peu moins confiant car sont des propositions
            
            # Gérer les accords/désaccords pattern-embedding si la relation vient d'un pattern et a été validée sémantiquement
            if rel.get('embedding_validation_status') == 'ACCORD_FORT':
                current_score *= self.config["pattern_embedding_accord_score"]
            elif rel.get('embedding_validation_status') == 'DESACCORD_FAIBLE':
                current_score *= self.config["pattern_embedding_discord_score"]

            # Assurer que le score ne dépasse pas 1.0
            rel['final_score'] = min(1.0, current_score)
            
            # Si la relation existe déjà, prendre la meilleure (score le plus élevé)
            if child_parent_key in combined_relations:
                if rel['final_score'] > combined_relations[child_parent_key]['final_score']:
                    combined_relations[child_parent_key] = rel
            else:
                combined_relations[child_parent_key] = rel

        # Filtrer par le seuil d'acceptation final
        final_filtered_relations = [
            rel for rel in combined_relations.values() 
            if rel['final_score'] >= self.config["acceptance_threshold"]
        ]

        return final_filtered_relations

    def run_validation_sample(self,
                              relations: List[Dict[str, Any]],
                              sample_size: int = 100) -> Dict[str, Any]:
        """
        Génère un échantillon de relations pour validation manuelle.

        Args:
            relations: Liste des relations hiérarchiques
            sample_size: Nombre d'exemples

        Returns:
            Dictionnaire des relations sélectionnées
        """
        import random
        sample = random.sample(relations, min(sample_size, len(relations)))
        
        # Calculer des métriques pour l'échantillon si nécessaire
        sample_metrics = {
            'sample_size': len(sample),
            'avg_final_score': np.mean([r['final_score'] for r in sample]) if sample else 0.0,
            'source_distribution': Counter([r['source'] for r in sample])
        }
        
        logger.info(f"Échantillon de validation généré : {sample_metrics['sample_size']} relations.")
        
        return {
            'sample': sample,
            'metrics': sample_metrics
        }