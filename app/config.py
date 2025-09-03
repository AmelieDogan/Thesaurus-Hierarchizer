#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de configuration pour l'application de hiérarchisation de thésaurus musical baroque.

Ce module centralise tous les paramètres configurables de l'application, y compris :
- La configuration du système de logging.
- Les seuils et paramètres spécifiques à chaque phase du pipeline de traitement.
- Les listes de mots-outils et autres constantes globales.

L'objectif est de rendre la configuration de l'application transparente et facilement
modulable sans modifier le code métier principal.
"""

import logging
import logging.config
import os

from typing import List, Dict, Any

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE = "app.log"

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'colored': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'log_colors': {
                'DEBUG':    'cyan',
                'INFO':     'blue',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            },
        },
        'file': {
            'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },

    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'colored',
            'level': LOG_LEVEL,
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': LOG_FILE,
            'formatter': 'file',
            'level': LOG_LEVEL,
        },
    },

    'root': {
        'handlers': ['console', 'file'],
        'level': LOG_LEVEL,
    },
}


def setup_logging():
    """
    Configure le système de logging de l'application en utilisant la
    configuration définie dans LOGGING_CONFIG.

    Cette fonction doit être appelée une seule fois au démarrage de l'application.
    Elle met en place des handlers pour la console (avec couleurs) et pour un fichier de log.
    Le niveau de log est contrôlé par la variable d'environnement LOG_LEVEL (par défaut DEBUG).
    """
    logging.config.dictConfig(LOGGING_CONFIG)
    
def load_default_config() -> Dict[str, Any]:
    """
    Charge la configuration par défaut de l'application.

    Tous les paramètres configurables du pipeline de hiérarchisation sont
    centralisés ici. Chaque clé du dictionnaire correspond à un paramètre
    utilisé dans une phase spécifique ou globalement par l'orchestrateur.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant l'ensemble des paramètres de configuration.
            Les clés et leurs descriptions sont les suivantes :

            * **output_dir** (str): Répertoire où seront sauvegardées toutes les sorties de
                l'application (logs, relations générées, statistiques, etc.).
            * **uri_base** (str): URI de base utilisée pour la génération des URIs des
                nouveaux termes dans le thésaurus, notamment pour les concepts générés.

            * **min_frequency_for_candidate** (int): Fréquence minimale d'apparition d'un mot
                dans l'ensemble du corpus pour qu'il soit considéré comme un candidat potentiel
                pour un parent lexical ou une entité significative (Phase 2).

            * **min_substring_length** (int): Longueur minimale des sous-chaînes communes à deux
                termes pour qu'elles soient prises en compte dans le calcul de similarité lexicale.
                Permet de filtrer les correspondances trop courtes et peu significatives (Phase 3).
            * **jaccard_threshold** (float): Seuil minimal de similarité de Jaccard entre deux
                termes pour considérer qu'une relation lexicale de proximité existe entre eux.
                Une valeur plus élevée indique une similarité plus stricte (Phase 3).
            * **min_family_size** (int): Taille minimale qu'une "famille lexicale" doit atteindre
                pour être considérée comme pertinente. Une famille est un groupe de termes
                fortement liés lexicalement (Phase 3).
            * **max_edit_distance** (int): Distance d'édition (Levenshtein) maximale autorisée
                entre deux termes pour les regrouper. Permet de tolérer de légères variations
                orthographiques, fautes de frappe ou flexions (ex: singulier/pluriel) (Phase 3).

            * **embedding_model_path** (str): Chemin ou nom du modèle SentenceTransformer à utiliser
                pour générer les embeddings sémantiques des termes. Ce modèle est utilisé pour
                capturer le sens contextuel des termes (Phase 4).
            * **abstract_similarity_threshold** (float): Seuil de similarité cosinus pour identifier
                des relations sémantiques entre des concepts considérés comme 'abstraits' ou moins
                spécifiques. Utilisé pour les relations de type 'broader' ou 'related' (Phase 4).
            * **validation_similarity_threshold** (float): Seuil de similarité cosinus utilisé pour la
                validation des ambiguïtés ou pour confirmer des relations sémantiques fortes entre
                termes. Typiquement plus élevé pour une confirmation rigoureuse (Phase 4).
            * **family_similarity_threshold** (float): Seuil de similarité cosinus pour enrichir des
                familles conceptuelles existantes ou pour regrouper de nouveaux termes dans des
                clusters sémantiques. Aide à la découverte de termes sémantiquement proches (Phase 4).
            * **clustering_n_clusters** (int | None): Nombre de clusters à former lors de l'agglomération
                sémantique. Si None, le clustering sera basé sur 'clustering_distance_threshold' (Phase 4).
            * **clustering_distance_threshold** (float): Seuil de distance (1 - similarité cosinus) pour
                l'agglomération hiérarchique si 'clustering_n_clusters' est None. Les clusters sont formés
                en regroupant les points dont la distance est inférieure à ce seuil (Phase 4).

            * **min_children_for_new_term** (int): Nombre minimal d'enfants qu'un nouveau terme générique
                (auto-généré ou proposé) doit avoir pour être considéré comme un parent valide dans la
                hiérarchie finale (Phase 5).
            * **normalization_method** (str): Méthode de normalisation des labels appliquée avant certaines
                comparaisons ou la génération de termes (ex: "accent_removal" pour supprimer les accents) (Phase 5).

            * **max_parents_per_term** (int): Nombre maximum de parents qu'un même terme peut avoir dans la
                hiérarchie finale. Permet de contrôler la poly-hiérarchie (Phase 6).
            * **enable_poly_hierarchy** (bool): Indicateur booléen pour autoriser ou non la poly-hiérarchie.
                Si False, chaque terme n'aura qu'un seul parent assigné (Phase 6).

            * **pattern_exact_score** (float): Score attribué à une relation détectée via une
                correspondance exacte de pattern lexical (très haute confiance) (Paramètre de Scoring).
            * **pattern_embedding_accord_score** (float): Multiplicateur de score appliqué quand une
                relation détectée par pattern lexical est fortement validée ou renforcée par l'analyse
                sémantique par embeddings (Paramètre de Scoring).
            * **pattern_only_score** (float): Score attribué aux relations basées uniquement sur des
                patterns lexicaux (sans validation sémantique forte), indiquant une bonne confiance
                (Paramètre de Scoring).
            * **embedding_only_score** (float): Score attribué aux relations basées uniquement sur les
                embeddings sémantiques (sans un pattern lexical clair), indiquant une confiance
                légèrement moindre que les patterns seuls (Paramètre de Scoring).
            * **pattern_embedding_discord_score** (float): Multiplicateur de score appliqué quand il y a
                un désaccord faible entre une relation détectée par pattern lexical et l'analyse sémantique.
                Indique une réduction de confiance (Paramètre de Scoring).
            * **acceptance_threshold** (float): Seuil d'acceptation global final pour filtrer les relations
                hiérarchiques proposées. Seules les relations dont le 'final_score' est supérieur ou égal
                à ce seuil seront conservées dans la hiérarchie finale (Paramètre de Scoring).
    """
    return {
        # ======================================================================
        # Paramètres Généraux
        # ======================================================================
        "output_dir": "output",
        "uri_base": "http://data.cmbv.org/vocabulary/",
        # ======================================================================
        # Phase 2: Détection de Patterns Lexicaux (LexicalPatternDetector)
        # ======================================================================
        "min_frequency_for_candidate": 3,
        # ======================================================================
        # Phase 3: Analyse de Similarité Lexicale (LexicalSimilarityAnalyzer)
        # ======================================================================
        "min_substring_length": 4, 
        "jaccard_threshold": 0.6, 
        "min_family_size": 2, 
        "max_edit_distance": 2, 
        # ======================================================================
        # Phase 4: Analyse Sémantique par Embeddings (SemanticEmbeddingAnalyzer)
        # ======================================================================
        "embedding_model_path": "dangvantuan/sentence-camembert-base",
        "context_window_size": 50,
        "pattern_min_frequency": 3,
        "graph_walk_length": 10,
        "zone_coherence_threshold": 0.75,
        "analogy_similarity_threshold": 0.70,
        "abstract_similarity_threshold": 0.65,
        "family_similarity_threshold": 0.60,
        # ======================================================================
        # Phase 5: Génération de Candidats Parents (ParentCandidateGenerator)
        # ======================================================================
        "min_children_for_new_term": 3,
        "normalization_method": "accent_removal",
        # ======================================================================
        # Phase 6: Optimisation Hiérarchique (HierarchyOptimizer)
        # ======================================================================
        "max_parents_per_term": 3, 
        "enable_poly_hierarchy": True, 
        # ======================================================================
        # Paramètres de Scoring pour l'Orchestrateur (main_orchestrator)
        # Ces scores sont utilisés pour combiner et évaluer les différentes sources de relations.
        # ======================================================================
        "pattern_exact_score": 1.0, 
        "pattern_embedding_accord_score": 1.0, 
        "pattern_only_score": 0.9, 
        "embedding_only_score": 0.8, 
        "pattern_embedding_discord_score": 0.5, 
        "acceptance_threshold": 0.8,
    }

def get_default_stop_words() -> List[str]:
    """
    Retourne la liste des mots-outils par défaut pour le traitement du thésaurus.

    Ces mots sont généralement des articles, prépositions, conjonctions, ou des
    termes très génériques spécifiques au domaine qui n'apportent pas de
    valeur discriminante pour la détection de relations hiérarchiques.

    Returns:
        List[str]: Liste des mots-outils français à exclure des analyses.
    """
    return [
        "à", "de", "du", "des", "le", "la", "les", "un", "une", 
        "et", "ou", "pour", "avec", "sans", "sur", "sous", 
        "dans", "par", "en", "au", "aux", "ce", "cette", "ces",
        "d", "l", "qu", "que", "qui", "dont", "où", "si", "mais",
        "car", "donc", "or", "ni", "soit", "comme", "entre", "vers",
        "chez", "depuis", "pendant", "après", "avant", "selon", "malgré",
        "iii", "jacques", "musical", "majeur", "petit", "anciens",
        "règle", "étranger", "regard", "simple", "paul", "jour", 
        "compassion","plusieurs", "editeur", "relations", "non", 
        "mineur", "pierre", "haute", "plein", "grand", "grande", "baptiste", 
        "fonds", "piece", "cadence", "chasse", "jean", "contre", 
        "fragment", "tragi"
    ]

def get_word_min_length() -> int:
    """
    Retourne le nombre minimum de caractères qu'un mot doit avoir
    pour être considéré comme significatif dans les analyses lexicales.

    Ce seuil permet de filtrer les mots très courts qui sont rarement
    des descripteurs pertinents pour un thésaurus.

    Returns:
        int: Nombre minimum de lettres pour qu'un mot soit considéré.
    """
    return 3