"""
Module générant les fichiers de sortie.
"""

import os
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
import json
from .logger import get_logger
from .phase6_hierarchy_optimizer import OptimizationStats # Import pour les stats

logger = get_logger(__name__)

class OutputGenerator:
    def __init__(self, output_dir: str = "output"):
        """
        Initialise le générateur de sorties.

        Args:
            output_dir: Dossier de sortie pour les fichiers générés
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Générateur de sorties initialisé. Dossier de sortie: {self.output_dir}")

    def create_skos_relations(self,
                              df: pd.DataFrame,
                              relations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Ajoute les colonnes skos:broader, skos:narrower (URIs et labels) au DataFrame
        à partir des relations optimisées.

        Args:
            df: DataFrame original avec 'URI', 'preflabel'.
            relations: Liste des relations hiérarchiques finales (doivent contenir 'child_uri', 'parent_uri', 'child', 'parent').

        Returns:
            DataFrame enrichi avec les relations SKOS.
        """
        logger.info("Création des colonnes skos:broader et skos:narrower dans le DataFrame...")
        broader_uri_map = defaultdict(list)
        narrower_uri_map = defaultdict(list)
        broader_label_map = defaultdict(list)
        narrower_label_map = defaultdict(list)

        for rel in relations:
            child_uri = rel.get("child_uri")
            parent_uri = rel.get("parent_uri")
            child_label = rel.get("child")
            parent_label = rel.get("parent")

            if child_uri and parent_uri:
                broader_uri_map[child_uri].append(parent_uri)
                narrower_uri_map[parent_uri].append(child_uri)
                if child_label and parent_label:
                    broader_label_map[child_uri].append(parent_label)
                    narrower_label_map[parent_uri].append(child_label)
            else:
                logger.warning(f"Relation ignorée car manque URI: {rel}")

        df_enriched = df.copy()
        
        # Initialiser les nouvelles colonnes
        df_enriched["skos:broader"] = None
        df_enriched["skos:narrower"] = None
        df_enriched["skos:broader_label"] = None
        df_enriched["skos:narrower_label"] = None

        # Remplir les colonnes basées sur les URIs
        # Utiliser un dictionnaire pour un mapping rapide URI -> index du df
        uri_to_idx = {uri: idx for idx, uri in enumerate(df_enriched['URI'])}

        for uri, idx in uri_to_idx.items():
            if uri in broader_uri_map:
                df_enriched.at[idx, "skos:broader"] = ";".join(broader_uri_map[uri])
                df_enriched.at[idx, "skos:broader_label"] = "; ".join(broader_label_map[uri])
            if uri in narrower_uri_map:
                df_enriched.at[idx, "skos:narrower"] = ";".join(narrower_uri_map[uri])
                df_enriched.at[idx, "skos:narrower_label"] = "; ".join(narrower_label_map[uri])
        
        logger.info("Colonnes SKOS créées avec succès.")
        return df_enriched

    def generate_main_output(self, df_enriched: pd.DataFrame) -> None:
        """
        Génère le fichier de sortie principal du thésaurus (TSV).

        Args:
            df_enriched: DataFrame enrichi avec les relations SKOS.
        """
        output_path = os.path.join(self.output_dir, "thesaurus_output.tsv")
        df_enriched.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        logger.info(f"Fichier de sortie principal généré: {output_path}")

    def generate_candidate_report(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Génère un rapport sur les candidats parents proposés.

        Args:
            candidates: Liste des candidats parents proposés.
        """
        if not candidates:
            logger.info("Aucun candidat parent à rapporter.")
            return

        report_data = []
        for cand in candidates:
            report_data.append({
                "child_uri": cand.get("child_uri", ""),
                "child_label": cand.get("child", ""),
                "parent_uri": cand.get("parent_uri", ""),
                "parent_label": cand.get("parent", ""),
                "confidence": cand.get("confidence", 0.0),
                "type": cand.get("type", "unknown"),
                "source_phase": cand.get("source_phase", "unknown"),
                "parent_type": cand.get("parent_type", "unknown") # existing_parent ou candidate_parent
            })
        
        df_report = pd.DataFrame(report_data)
        output_path = os.path.join(self.output_dir, "candidate_report.tsv")
        df_report.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        logger.info(f"Rapport des candidats parents généré: {output_path}")

    def generate_decision_log(self, conflicts: List[Dict[str, Any]]) -> None:
        """
        Génère un log des décisions prises (conflits, etc.).

        Args:
            conflicts: Liste des conflits détectés.
        """
        output_path = os.path.join(self.output_dir, "decision_log.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conflicts, f, indent=4, ensure_ascii=False)
            logger.info(f"Log des décisions généré: {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la génération du log des décisions: {e}")

    def generate_summary_report(self, df: pd.DataFrame, 
                                relations: List[Dict[str, Any]],
                                opt_stats: OptimizationStats) -> None:
        """
        Génère un rapport de synthèse des métriques globales de la hiérarchie.

        Args:
            df: DataFrame enrichi.
            relations: Relations finales hiérarchiques.
            opt_stats: Statistiques d'optimisation de la hiérarchie.
        """
        logger.info("Génération du rapport de synthèse...")
        total_terms = len(df)
        
        # Compter les termes avec des parents ou enfants via les relations optimisées
        terms_with_parents = set(rel["child_uri"] for rel in relations if rel.get("child_uri"))
        terms_with_children = set(rel["parent_uri"] for rel in relations if rel.get("parent_uri"))
        
        total_relations = len(relations)

        summary_metrics = {
            "Total des termes dans le thésaurus": total_terms,
            "Termes avec au moins un parent (URI)": len(terms_with_parents),
            "Termes avec au moins un enfant (URI)": len(terms_with_children),
            "Nombre total de relations (optimisées)": total_relations,
            "Moyenne de parents par terme": round(total_relations / total_terms, 3) if total_terms > 0 else 0,
            
            "\n--- Statistiques d'optimisation ---": "",
            "Relations initiales (avant optimisation)": opt_stats.initial_relations_count,
            "Relations finales (après optimisation)": opt_stats.final_relations_count,
            "Relations redondantes supprimées": opt_stats.redundancies_removed,
            "Cycles détectés": opt_stats.cycles_detected,
            "Cycles résolus (relations supprimées)": opt_stats.cycles_resolved,
            "Ajustements de poly-hiérarchie": opt_stats.poly_hierarchy_adjustments,
            "Temps d'exécution de l'optimisation (s)": round(opt_stats.execution_time, 3)
        }
        
        output_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for key, value in summary_metrics.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Rapport de synthèse généré: {output_path}")

    def generate_all_outputs(self,
                             df: pd.DataFrame,
                             relations: List[Dict[str, Any]],
                             candidates: List[Dict[str, Any]],
                             conflicts: List[Dict[str, Any]],
                             optimization_stats: OptimizationStats) -> None:
        """
        Exécute toutes les étapes de sortie.

        Args:
            df: DataFrame enrichi avec les données initiales et les nouveaux termes.
            relations: Relations hiérarchiques finales (optimisées).
            candidates: Candidats générés (avant optimisation).
            conflicts: Conflits pattern/embedding détectés.
            optimization_stats: Statistiques de l'optimiseur de hiérarchie.
        """
        logger.info("Début de la génération de tous les fichiers de sortie...")
        
        # 1. Création des relations SKOS dans le DataFrame
        df_enriched = self.create_skos_relations(df, relations)
        
        # 2. Génération du fichier de sortie principal
        self.generate_main_output(df_enriched)
        
        # 3. Génération du rapport des candidats
        self.generate_candidate_report(candidates)
        
        # 4. Génération du log des décisions (conflits)
        self.generate_decision_log(conflicts)
        
        # 5. Génération du rapport de synthèse
        self.generate_summary_report(df, relations, optimization_stats) # df non enrichi pour total termes
        
        logger.info("Tous les fichiers de sortie ont été générés avec succès.")