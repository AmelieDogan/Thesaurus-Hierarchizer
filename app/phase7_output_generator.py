"""
Module des fichiers de sortie pour le thésaurus - Phase 7.

Ce module génére les sorties finales à savoir le thésaurus sous deux formats, une fichier
TSV compatible SKOS et un fichier XML RDF/SKOS, ainsi que le rapport donnant tous les
statistiques des différentes phases.
"""

from typing import Dict, List, Any
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import html
from datetime import datetime
import os
from pathlib import Path
import logging


class OutputGenerator:
    """
    Phase 7 : Génération des sorties finales du pipeline de hiérarchisation.
    
    Cette classe génère trois sorties principales :
    1. Fichier TSV SKOS enrichi avec hiérarchie
    2. Fichier XML RDF/SKOS standard 
    3. Rapport HTML de synthèse du pipeline
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialise le générateur de sorties.
        
        Args:
            config: Configuration du pipeline (doit contenir output_dir, uri_base)
            logger: Logger pour les messages
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Paramètres de génération
        self.output_dir = Path(config["output_dir"])
        self.uri_base = config["uri_base"]
        self.xml_language = config.get("xml_language", "fr")
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colonnes TSV SKOS standard (ordre préservé)
        self.tsv_columns = [
            "URI", "skos:prefLabel", "skos:definition", "rdf:type", "skos:altLabel", 
            "skos:hiddenLabel", "skos:narrower", "skos:broader", 
            "skos:scopeNote", "skos:changeNote"
        ]
        
        # Templates et styles pour le rapport HTML
        self.html_template = self._get_html_template()
        
    def generate_all_outputs(self, 
                           optimized_hierarchy: Dict[str, Any],
                           enriched_indexes: Dict[str, Any], 
                           processed_dataframe: pd.DataFrame,
                           pipeline_stats: Dict[str, Any]) -> Dict[str, str]:
        """
        Génère toutes les sorties finales du pipeline.
        
        Args:
            optimized_hierarchy: Hiérarchie optimisée de la phase 6
            enriched_indexes: Index enrichis avec uri_to_original_data
            processed_dataframe: DataFrame de la phase 1 (avec URIs ajoutés)
            pipeline_stats: Statistiques cumulées de toutes les phases
            
        Returns:
            Dict avec les chemins des fichiers générés : 
            {"tsv": path, "xml": path, "html": path}
        """
        self.logger.info("=== PHASE 7 : Génération des sorties finales ===")
        
        output_files = {}
        
        try:
            # Génération TSV
            tsv_path = self.generate_tsv_output(
                optimized_hierarchy, enriched_indexes, processed_dataframe
            )
            output_files["tsv"] = tsv_path
            
            # Génération XML RDF/SKOS
            xml_path = self.generate_rdf_xml_output(
                optimized_hierarchy, enriched_indexes, processed_dataframe
            )
            output_files["xml"] = xml_path
            
            # Génération rapport HTML
            html_path = self.generate_html_report(pipeline_stats, optimized_hierarchy)
            output_files["html"] = html_path
            
            self.logger.info("=== Phase 7 terminée avec succès ===")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des sorties : {str(e)}")
            raise
        
        return output_files

    def generate_tsv_output(self, 
                           optimized_hierarchy: Dict[str, Any],
                           enriched_indexes: Dict[str, Any],
                           processed_dataframe: pd.DataFrame) -> str:
        """
        Génère le fichier TSV SKOS enrichi avec la hiérarchie optimisée.
        
        Processus :
        1. Reconstituer le DataFrame complet depuis la hiérarchie
        2. Enrichir avec les relations broader/narrower calculées
        3. Ajouter changeNote pour les termes candidats
        4. Préserver toutes les données originales
        5. Exporter au format TSV
        
        Args:
            optimized_hierarchy: Structure des nœuds optimisée
            enriched_indexes: Index avec correspondances URI <-> données originales
            processed_dataframe: DataFrame original avec colonnes enrichies (phase 1)
            
        Returns:
            Chemin du fichier TSV généré
        """
        self.logger.info("Génération du fichier TSV SKOS enrichi")
        
        # Étape 1 : Créer un DataFrame complet à partir de la hiérarchie
        output_rows = []
        
        for uri, node_data in optimized_hierarchy["nodes"].items():
            row = self._create_tsv_row_from_node(uri, node_data, enriched_indexes, processed_dataframe)
            output_rows.append(row)
        
        # Étape 2 : Créer le DataFrame de sortie
        output_df = pd.DataFrame(output_rows, columns=self.tsv_columns)
        
        # Étape 3 : Trier par prefLabel pour lisibilité
        output_df = output_df.sort_values('skos:prefLabel')
        
        # Étape 4 : Sauvegarder
        output_path = self.output_dir / "thesaurus_hierarchized.tsv"
        output_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        
        self.logger.info(f"Fichier TSV généré : {output_path} ({len(output_df)} concepts)")
        return str(output_path)

    def _create_tsv_row_from_node(self, 
                                 uri: str, 
                                 node_data: Dict[str, Any],
                                 enriched_indexes: Dict[str, Any],
                                 processed_dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Crée une ligne TSV à partir d'un nœud de la hiérarchie optimisée.
        
        Args:
            uri: URI du concept
            node_data: Données du nœud depuis optimized_hierarchy
            enriched_indexes: Index pour retrouver données originales
            processed_dataframe: DataFrame original pour données manquantes
            
        Returns:
            Dictionnaire représentant une ligne TSV complète
        """
        # Récupérer les données originales
        original_data = self._get_original_data_for_uri(uri, enriched_indexes, processed_dataframe)
        
        # Construire la ligne de sortie
        row = {}

        row["URI"] = original_data.get("URI", "") or node_data.get("uri")
        
        # prefLabel : utiliser le preflabel nettoyé de la hiérarchie ou original
        row["skos:prefLabel"] = original_data.get("skos:prefLabel", "") or node_data.get("preflabel")
        
        # Données originales préservées
        row["skos:definition"] = original_data.get("skos:definition", "")
        row["rdf:type"] = original_data.get("rdf:type", "skos:Concept") or "skos:Concept"
        row["skos:altLabel"] = original_data.get("skos:altLabel", "")
        row["skos:hiddenLabel"] = original_data.get("skos:hiddenLabel", "")
        row["skos:scopeNote"] = original_data.get("skos:scopeNote", "")
        
        # Relations hiérarchiques calculées
        row["skos:broader"] = self._format_uri_list(node_data.get("parents", []))
        row["skos:narrower"] = self._format_uri_list(node_data.get("children", []))
        
        # changeNote enrichie
        row["skos:changeNote"] = original_data.get("skos:changeNote", "") or "Terme créé lors de la hiérarchisation semi-automatique"
        
        return row

    def _get_original_data_for_uri(self, 
                                  uri: str, 
                                  enriched_indexes: Dict[str, Any],
                                  processed_dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Récupère les données originales pour un URI donné.
        
        Recherche d'abord dans enriched_indexes["uri_to_original_data"],
        puis en fallback dans le processed_dataframe.
        
        Args:
            uri: URI du concept
            enriched_indexes: Index enrichis
            processed_dataframe: DataFrame de fallback
            
        Returns:
            Dictionnaire des données originales
        """
        # Méthode 1 : via enriched_indexes
        if "uri_to_original_data" in enriched_indexes:
            original_data = enriched_indexes["uri_to_original_data"].get(uri)
            if original_data:
                return original_data
        
        # Méthode 2 : recherche dans le DataFrame par URI
        if "URI" in processed_dataframe.columns:
            matching_rows = processed_dataframe[processed_dataframe["URI"] == uri]
            if not matching_rows.empty:
                return matching_rows.iloc[0].to_dict()
        
        # Méthode 3 : recherche par préfLabel via enriched_indexes
        if uri in enriched_indexes.get("uri_to_preflabel", {}):
            preflabel = enriched_indexes["uri_to_preflabel"][uri]
            # Rechercher dans le DataFrame par preflabel_clean
            if "preflabel_clean" in processed_dataframe.columns:
                matching_rows = processed_dataframe[processed_dataframe["preflabel_clean"] == preflabel]
                if not matching_rows.empty:
                    return matching_rows.iloc[0].to_dict()
        
        self.logger.warning(f"Données originales non trouvées pour URI {uri}")
        return {}

    def _format_uri_list(self, uri_list: List[str]) -> str:
        """
        Formate une liste d'URIs au format SKOS (séparateur " ## ").
        
        Args:
            uri_list: Liste des URIs
            
        Returns:
            String formatée pour SKOS ou chaîne vide
        """
        if not uri_list:
            return ""
        
        # Éliminer les doublons tout en préservant l'ordre
        unique_uris = []
        seen = set()
        for uri in uri_list:
            if uri not in seen:
                unique_uris.append(uri)
                seen.add(uri)
        
        return " ## ".join(unique_uris)

    def generate_rdf_xml_output(self, 
                               optimized_hierarchy: Dict[str, Any],
                               enriched_indexes: Dict[str, Any], 
                               processed_dataframe: pd.DataFrame) -> str:
        """
        Génère le fichier XML RDF/SKOS standard de la hiérarchie optimisée.
        
        Structure générée :
        - En-tête RDF avec namespaces
        - ConceptScheme principal  
        - Un skos:Concept par terme avec toutes les propriétés
        
        Args:
            optimized_hierarchy: Structure des nœuds optimisée
            enriched_indexes: Index pour retrouver données originales
            processed_dataframe: DataFrame original
            
        Returns:
            Chemin du fichier XML généré
        """
        self.logger.info("Génération du fichier XML RDF/SKOS")
        
        # Créer l'élément racine RDF avec namespaces
        rdf_root = ET.Element("rdf:RDF")
        rdf_root.set("xmlns:rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        rdf_root.set("xmlns:skos", "http://www.w3.org/2004/02/skos/core#")
        rdf_root.set("xmlns:dc", "http://purl.org/dc/elements/1.1/")
        
        # Ajouter le ConceptScheme
        concept_scheme = self._create_concept_scheme()
        rdf_root.append(concept_scheme)
        
        # Ajouter chaque concept
        for uri, node_data in optimized_hierarchy["nodes"].items():
            concept_element = self._create_concept_element(
                uri, node_data, enriched_indexes, processed_dataframe
            )
            rdf_root.append(concept_element)
        
        # Formatter et sauvegarder le XML
        xml_str = self._format_xml(rdf_root)
        output_path = self.output_dir / "thesaurus_hierarchized.rdf"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        self.logger.info(f"Fichier XML généré : {output_path} ({len(optimized_hierarchy['nodes'])} concepts)")
        return str(output_path)

    def _create_concept_scheme(self) -> ET.Element:
        """
        Crée l'élément ConceptScheme pour le fichier RDF.
        
        Returns:
            Élément XML ConceptScheme
        """
        scheme = ET.Element("skos:ConceptScheme")
        scheme.set("rdf:about", self.uri_base)
        
        # Métadonnées du schéma
        title = ET.SubElement(scheme, "dc:title")
        title.text = "Thésaurus hiérarchisé automatiquement"
        
        creator = ET.SubElement(scheme, "dc:creator")  
        creator.text = "Pipeline de hiérarchisation automatique"
        
        date = ET.SubElement(scheme, "dc:date")
        date.text = datetime.now().isoformat()[:10]  # Format YYYY-MM-DD
        
        return scheme

    def _create_concept_element(self, 
                               uri: str,
                               node_data: Dict[str, Any],
                               enriched_indexes: Dict[str, Any],
                               processed_dataframe: pd.DataFrame) -> ET.Element:
        """
        Crée un élément skos:Concept pour le XML RDF.
        
        Args:
            uri: URI du concept
            node_data: Données du nœud
            enriched_indexes: Index pour retrouver données originales
            processed_dataframe: DataFrame original
            
        Returns:
            Élément XML skos:Concept complet
        """
        concept = ET.Element("skos:Concept")
        concept.set("rdf:about", uri)
        
        # Référence au schéma conceptuel
        in_scheme = ET.SubElement(concept, "skos:inScheme")
        in_scheme.set("rdf:resource", self.uri_base)
        
        # Récupérer les données originales
        original_data = self._get_original_data_for_uri(uri, enriched_indexes, processed_dataframe)
        
        # prefLabel
        preflabel = original_data.get("skos:prefLabel", "") or node_data.get("preflabel")
        if preflabel:
            pref_elem = ET.SubElement(concept, "skos:prefLabel")
            pref_elem.set("xml:lang", self.xml_language)
            pref_elem.text = self._clean_text_for_xml(preflabel)
        
        # definition  
        definition = original_data.get("skos:definition", "")
        if definition:
            def_elem = ET.SubElement(concept, "skos:definition")
            def_elem.set("xml:lang", self.xml_language)
            def_elem.text = self._clean_text_for_xml(definition)
        
        # altLabel (peut être multiple)
        alt_labels = original_data.get("skos:altLabel", "")
        if alt_labels:
            for alt_label in self._split_multi_value(alt_labels):
                alt_elem = ET.SubElement(concept, "skos:altLabel")
                alt_elem.set("xml:lang", self.xml_language)
                alt_elem.text = self._clean_text_for_xml(alt_label)
        
        # Relations broader (parents)
        for parent_uri in node_data.get("parents", []):
            broader_elem = ET.SubElement(concept, "skos:broader")
            broader_elem.set("rdf:resource", parent_uri)
        
        # Relations narrower (enfants)
        for child_uri in node_data.get("children", []):
            narrower_elem = ET.SubElement(concept, "skos:narrower")
            narrower_elem.set("rdf:resource", child_uri)
        
        # scopeNote si présente
        scope_note = original_data.get("skos:scopeNote", "")
        if scope_note:
            scope_elem = ET.SubElement(concept, "skos:scopeNote")
            scope_elem.set("xml:lang", self.xml_language)
            scope_elem.text = self._clean_text_for_xml(scope_note)
        
        # changeNote enrichie
        change_note = original_data.get("skos:changeNote", "") or "Terme créé lors de la hiérarchisation semi-automatique"
        if change_note:
            change_elem = ET.SubElement(concept, "skos:changeNote")
            change_elem.set("xml:lang", self.xml_language)
            change_elem.text = self._clean_text_for_xml(change_note)
        
        return concept

    def _clean_text_for_xml(self, text: str) -> str:
        """
        Nettoie et échappe le texte pour inclusion dans XML.
        
        Args:
            text: Texte à nettoyer
            
        Returns:
            Texte échappé pour XML
        """
        if not text or pd.isna(text):
            return ""
        
        # Échapper les caractères XML spéciaux
        text = str(text).strip()
        text = html.escape(text, quote=False)
        
        # Nettoyer les caractères de contrôle non valides en XML
        valid_chars = []
        for c in text:
            if ord(c) >= 32 or c in '\t\n\r':
                valid_chars.append(c)
        
        return ''.join(valid_chars)

    def _split_multi_value(self, value: str) -> List[str]:
        """
        Divise une valeur multi (format " ## ") en liste.
        
        Args:
            value: Valeur potentiellement multiple
            
        Returns:
            Liste des valeurs individuelles
        """
        if not value or pd.isna(value):
            return []
        
        return [v.strip() for v in str(value).split("##") if v.strip()]

    def _format_xml(self, root_element: ET.Element) -> str:
        """
        Formate l'élément XML avec indentation pour lisibilité.
        
        Args:
            root_element: Élément racine XML
            
        Returns:
            String XML formatée avec en-tête
        """
        # Créer la string XML brute
        rough_string = ET.tostring(root_element, encoding='unicode')
        
        # Parser et formater avec minidom
        parsed = minidom.parseString(rough_string)
        formatted = parsed.toprettyxml(indent="  ", encoding=None)
        
        # Nettoyer l'en-tête par défaut et ajouter le nôtre
        lines = formatted.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        # Ajouter notre en-tête XML
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
        
        return xml_header + '\n'.join(line for line in lines if line.strip())

    def generate_html_report(self, 
                            pipeline_stats: Dict[str, Any],
                            optimized_hierarchy: Dict[str, Any]) -> str:
        """
        Génère le rapport HTML de synthèse du pipeline complet.
        
        Args:
            pipeline_stats: Statistiques cumulées de toutes les phases
            optimized_hierarchy: Hiérarchie finale pour métriques de qualité
            
        Returns:
            Chemin du fichier HTML généré
        """
        self.logger.info("Génération du rapport HTML de synthèse")
        
        # Calculer les métriques de qualité finale
        final_metrics = self._calculate_final_hierarchy_metrics(optimized_hierarchy)
        
        # Générer le contenu HTML
        html_content = self.html_template.format(
            generation_date=datetime.now().strftime("%d/%m/%Y à %H:%M:%S"),
            executive_summary=self._generate_executive_summary(pipeline_stats, final_metrics),
            phase1_section=self._generate_phase_section("Phase 1 - Préparation des données", pipeline_stats.get("phase1", {})),
            phase2_section=self._generate_phase_section("Phase 2 - Détection de patterns", pipeline_stats.get("phase2", {})),
            phase3_section=self._generate_phase_section("Phase 3 - Similarité lexicale", pipeline_stats.get("phase3", {})),
            phase4_section=self._generate_phase_section("Phase 4 - Analyse sémantique", pipeline_stats.get("phase4", {})),
            phase5_section=self._generate_phase_section("Phase 5 - Construction hiérarchique", pipeline_stats.get("phase5", {})),
            phase6_section=self._generate_phase_section("Phase 6 - Optimisation", pipeline_stats.get("phase6", {})),
            final_metrics_section=self._generate_final_metrics_section(final_metrics),
            recommendations_section=self._generate_recommendations(pipeline_stats, final_metrics)
        )
        
        # Sauvegarder le fichier HTML
        output_path = self.output_dir / "hierarchization_report.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Rapport HTML généré : {output_path}")
        return str(output_path)

    def _calculate_final_hierarchy_metrics(self, optimized_hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcule les métriques de qualité de la hiérarchie finale.
        
        Args:
            optimized_hierarchy: Structure de la hiérarchie optimisée
            
        Returns:
            Dictionnaire des métriques calculées
        """
        nodes = optimized_hierarchy.get("nodes", {})
        
        if not nodes:
            return {"error": "Aucun nœud dans la hiérarchie"}
        
        metrics = {
            "total_concepts": len(nodes),
            "root_concepts": 0,        # Pas de parents
            "leaf_concepts": 0,        # Pas d'enfants  
            "orphan_concepts": 0,      # Ni parents ni enfants
            "total_relations": 0,
            "avg_parents_per_concept": 0.0,
            "avg_children_per_concept": 0.0,
            "max_depth": 0,
            "hierarchy_coverage": 0.0,  # % de concepts avec au moins 1 relation
            "poly_hierarchy_rate": 0.0  # % de concepts avec >1 parent
        }
        
        total_parents = 0
        total_children = 0
        concepts_with_multiple_parents = 0
        concepts_with_relations = 0
        
        for uri, node_data in nodes.items():
            parents = node_data.get("parents", [])
            children = node_data.get("children", [])
            
            parent_count = len(parents)  
            child_count = len(children)
            
            total_parents += parent_count
            total_children += child_count
            metrics["total_relations"] += child_count  # Compter par les enfants
            
            # Classification des nœuds
            if parent_count == 0 and child_count == 0:
                metrics["orphan_concepts"] += 1
            elif parent_count == 0:
                metrics["root_concepts"] += 1
            elif child_count == 0:
                metrics["leaf_concepts"] += 1
                
            # Poly-hiérarchie
            if parent_count > 1:
                concepts_with_multiple_parents += 1
                
            # Couverture hiérarchique
            if parent_count > 0 or child_count > 0:
                concepts_with_relations += 1
        
        # Calculer les moyennes et pourcentages
        total_concepts = metrics["total_concepts"]
        
        if total_concepts > 0:
            metrics["avg_parents_per_concept"] = round(total_parents / total_concepts, 2)
            metrics["avg_children_per_concept"] = round(total_children / total_concepts, 2)
            metrics["hierarchy_coverage"] = round((concepts_with_relations / total_concepts) * 100, 2)
            metrics["poly_hierarchy_rate"] = round((concepts_with_multiple_parents / total_concepts) * 100, 2)
        
        # Calculer la profondeur maximale (approximation simple)
        metrics["max_depth"] = self._estimate_max_depth(nodes)
        
        return metrics

    def _estimate_max_depth(self, nodes: Dict[str, Any]) -> int:
        """
        Estime la profondeur maximale de la hiérarchie.
        
        Méthode simple : partir des racines et suivre les chemins.
        
        Args:
            nodes: Dictionnaire des nœuds
            
        Returns:
            Profondeur maximale estimée
        """
        # Identifier les nœuds racines
        root_nodes = [uri for uri, data in nodes.items() if not data.get("parents", [])]
        
        if not root_nodes:
            return 0
        
        max_depth = 0
        
        # Pour chaque racine, calculer la profondeur maximale
        for root_uri in root_nodes:
            depth = self._calculate_depth_from_node(root_uri, nodes, visited=set())
            max_depth = max(max_depth, depth)
        
        return max_depth

    def _calculate_depth_from_node(self, uri: str, nodes: Dict[str, Any], visited: set) -> int:
        """
        Calcule récursivement la profondeur depuis un nœud donné.
        
        Args:
            uri: URI du nœud courant
            nodes: Dictionnaire des nœuds
            visited: Set des nœuds visités (éviter cycles)
            
        Returns:
            Profondeur maximale depuis ce nœud
        """
        if uri in visited or uri not in nodes:
            return 0
        
        visited.add(uri)
        
        children = nodes[uri].get("children", [])
        if not children:
            visited.remove(uri)
            return 1
        
        max_child_depth = 0
        for child_uri in children:
            child_depth = self._calculate_depth_from_node(child_uri, nodes, visited)
            max_child_depth = max(max_child_depth, child_depth)
        
        visited.remove(uri)
        return 1 + max_child_depth

    def _generate_executive_summary(self, 
                                   pipeline_stats: Dict[str, Any], 
                                   final_metrics: Dict[str, Any]) -> str:
        """
        Génère le résumé exécutif du rapport HTML.
        
        Args:
            pipeline_stats: Statistiques du pipeline
            final_metrics: Métriques de la hiérarchie finale
            
        Returns:
            HTML du résumé exécutif
        """
        total_concepts = final_metrics.get("total_concepts", 0)
        total_relations = final_metrics.get("total_relations", 0)
        coverage = final_metrics.get("hierarchy_coverage", 0)
        
        # Récupérer les statistiques de la phase 1 pour comparaison
        phase1_stats = pipeline_stats.get("phase1", {})
        initial_terms = phase1_stats.get("total_terms", 0)
        initial_relations = phase1_stats.get("total_existing_broader_relations", 0)
        
        return f"""
        <div class="summary-box">
            <h3>🎯 Résultats du Pipeline</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-value">{total_concepts}</span>
                    <span class="metric-label">Concepts traités</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{total_relations}</span>
                    <span class="metric-label">Relations hiérarchiques</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{coverage}%</span>
                    <span class="metric-label">Couverture hiérarchique</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{final_metrics.get('max_depth', 0)}</span>
                    <span class="metric-label">Profondeur max.</span>
                </div>
            </div>
            <p class="summary-text">
                Le pipeline a traité {total_concepts} concepts et généré {total_relations} relations hiérarchiques, 
                atteignant une couverture de {coverage}%. 
                {f"Amélioration significative par rapport aux {initial_relations} relations initiales." if total_relations > initial_relations else ""}
            </p>
        </div>
        """

    def _generate_phase_section(self, title: str, stats: Dict[str, Any]) -> str:
        """
        Génère une section HTML pour une phase du pipeline.
        
        Args:
            title: Titre de la phase
            stats: Statistiques de la phase
            
        Returns:
            HTML de la section
        """
        if not stats:
            return f"""
            <div class="phase-section">
                <h3>{title}</h3>
                <p>Aucune statistique disponible pour cette phase.</p>
            </div>
            """
        
        stats_html = ""
        for key, value in stats.items():
            if isinstance(value, dict):
                # Sous-sections pour les statistiques complexes
                stats_html += f"<h4>{key.replace('_', ' ').title()}</h4>"
                for sub_key, sub_value in value.items():
                    stats_html += f"<p><strong>{sub_key.replace('_', ' ').title()}:</strong> {sub_value}</p>"
            else:
                stats_html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
        
        return f"""
        <div class="phase-section">
            <h3>{title}</h3>
            <div class="stats-content">
                {stats_html}
            </div>
        </div>
        """

    def _generate_final_metrics_section(self, metrics: Dict[str, Any]) -> str:
        """
        Génère la section des métriques finales.
        
        Args:
            metrics: Métriques calculées de la hiérarchie
            
        Returns:
            HTML de la section métriques
        """
        if "error" in metrics:
            return f"""
            <div class="phase-section error">
                <h3>Métriques de la Hiérarchie Finale</h3>
                <p class="error-message">{metrics['error']}</p>
            </div>
            """
        
        return f"""
        <div class="phase-section">
            <h3>📊 Métriques de la Hiérarchie Finale</h3>
            <div class="metrics-detailed">
                <div class="metric-row">
                    <span class="metric-name">Concepts totaux:</span>
                    <span class="metric-val">{metrics.get('total_concepts', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Concepts racines:</span>
                    <span class="metric-val">{metrics.get('root_concepts', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Concepts feuilles:</span>
                    <span class="metric-val">{metrics.get('leaf_concepts', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Concepts orphelins:</span>
                    <span class="metric-val">{metrics.get('orphan_concepts', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Relations hiérarchiques:</span>
                    <span class="metric-val">{metrics.get('total_relations', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Profondeur maximale:</span>
                    <span class="metric-val">{metrics.get('max_depth', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Couverture hiérarchique:</span>
                    <span class="metric-val">{metrics.get('hierarchy_coverage', 0)}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Taux de poly-hiérarchie:</span>
                    <span class="metric-val">{metrics.get('poly_hierarchy_rate', 0)}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Moyenne parents/concept:</span>
                    <span class="metric-val">{metrics.get('avg_parents_per_concept', 0)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Moyenne enfants/concept:</span>
                    <span class="metric-val">{metrics.get('avg_children_per_concept', 0)}</span>
                </div>
            </div>
        </div>
        """

    def _generate_recommendations(self, 
                                pipeline_stats: Dict[str, Any], 
                                final_metrics: Dict[str, Any]) -> str:
        """
        Génère des recommandations basées sur les métriques.
        
        Args:
            pipeline_stats: Statistiques du pipeline
            final_metrics: Métriques finales
            
        Returns:
            HTML des recommandations
        """
        recommendations = []
        
        # Analyser la couverture hiérarchique
        coverage = final_metrics.get('hierarchy_coverage', 0)
        if coverage < 50:
            recommendations.append("⚠️ Couverture hiérarchique faible (<50%). Considérer l'ajustement des seuils de similarité.")
        elif coverage > 90:
            recommendations.append("✅ Excellente couverture hiérarchique (>90%).")
        
        # Analyser les concepts orphelins
        orphans = final_metrics.get('orphan_concepts', 0)
        total = final_metrics.get('total_concepts', 1)
        orphan_rate = (orphans / total) * 100 if total > 0 else 0
        
        if orphan_rate > 30:
            recommendations.append("⚠️ Taux élevé de concepts orphelins (>30%). Réviser les stratégies de détection de relations.")
        elif orphan_rate < 10:
            recommendations.append("✅ Faible taux de concepts orphelins (<10%).")
        
        # Analyser la profondeur
        max_depth = final_metrics.get('max_depth', 0)
        if max_depth < 3:
            recommendations.append("⚠️ Hiérarchie peu profonde. Possibles relations manquées.")
        elif max_depth > 8:
            recommendations.append("⚠️ Hiérarchie très profonde. Vérifier les sur-spécialisations.")
        else:
            recommendations.append("✅ Profondeur hiérarchique équilibrée.")
        
        # Analyser la poly-hiérarchie
        poly_rate = final_metrics.get('poly_hierarchy_rate', 0)
        if poly_rate > 20:
            recommendations.append("ℹ️ Taux élevé de poly-hiérarchie. Normal pour des domaines complexes.")
        
        # Recommandations génériques si aucune spécifique
        if not recommendations:
            recommendations.append("✅ Les métriques de la hiérarchie semblent équilibrées.")
        
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        
        return f"""
        <div class="phase-section">
            <h3>💡 Recommandations</h3>
            <ul class="recommendations">
                {recommendations_html}
            </ul>
            <div class="next-steps">
                <h4>Prochaines étapes suggérées :</h4>
                <ol>
                    <li>Valider manuellement un échantillon des relations générées</li>
                    <li>Intégrer le thésaurus hiérarchisé dans votre système</li>
                    <li>Surveiller les performances sur les cas d'usage métier</li>
                    <li>Itérer sur la configuration si nécessaire</li>
                </ol>
            </div>
        </div>
        """

    def _get_html_template(self) -> str:
        """
        Retourne le template HTML pour le rapport.
        
        Returns:
            Template HTML avec placeholders
        """
        return """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Hiérarchisation - Pipeline Thésaurus</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .summary-box {{
            background: #f8f9ff;
            border: 1px solid #e1e5f2;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        
        .summary-box h3 {{
            margin-top: 0;
            color: #5a67d8;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric {{
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }}
        
        .metric-value {{
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}
        
        .metric-label {{
            display: block;
            font-size: 0.9em;
            color: #718096;
            margin-top: 5px;
        }}
        
        .summary-text {{
            color: #4a5568;
            font-size: 1.1em;
            margin-top: 20px;
        }}
        
        .phase-section {{
            margin-bottom: 30px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .phase-section h3 {{
            background: #edf2f7;
            margin: 0;
            padding: 15px 20px;
            color: #2d3748;
            font-size: 1.3em;
        }}
        
        .phase-section.error h3 {{
            background: #fed7d7;
            color: #c53030;
        }}
        
        .stats-content, .metrics-detailed {{
            padding: 20px;
        }}
        
        .stats-content p {{
            margin: 8px 0;
        }}
        
        .stats-content strong {{
            color: #2d3748;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f7fafc;
        }}
        
        .metric-row:last-child {{
            border-bottom: none;
        }}
        
        .metric-name {{
            color: #4a5568;
        }}
        
        .metric-val {{
            font-weight: bold;
            color: #2d3748;
        }}
        
        .recommendations {{
            list-style: none;
            padding: 0;
        }}
        
        .recommendations li {{
            padding: 10px;
            margin: 5px 0;
            background: #f7fafc;
            border-radius: 4px;
            border-left: 4px solid #4299e1;
        }}
        
        .next-steps {{
            margin-top: 25px;
            padding: 20px;
            background: #f0fff4;
            border-radius: 6px;
            border: 1px solid #9ae6b4;
        }}
        
        .next-steps h4 {{
            margin-top: 0;
            color: #2f855a;
        }}
        
        .next-steps ol {{
            color: #276749;
        }}
        
        .error-message {{
            color: #c53030;
            font-weight: bold;
            text-align: center;
            padding: 20px;
        }}
        
        .footer {{
            background: #f7fafc;
            padding: 20px;
            text-align: center;
            color: #718096;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .metric-row {{
                flex-direction: column;
            }}
            
            .metric-val {{
                text-align: right;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Rapport de Hiérarchisation</h1>
            <p class="subtitle">Pipeline de Construction Automatique de Thésaurus</p>
            <p class="subtitle">Généré le {generation_date}</p>
        </div>
        
        <div class="content">
            {executive_summary}
            
            <div class="pipeline-phases">
                {phase1_section}
                {phase2_section}
                {phase3_section}
                {phase4_section}
                {phase5_section}
                {phase6_section}
            </div>
            
            {final_metrics_section}
            {recommendations_section}
        </div>
        
        <div class="footer">
            <p>Ce rapport a été généré automatiquement par le Pipeline de Hiérarchisation de Thésaurus</p>
            <p>Version du pipeline : 1.0 | Phase 7 - OutputGenerator</p>
        </div>
    </div>
</body>
</html>"""