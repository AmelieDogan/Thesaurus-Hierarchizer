# Thesaurus-Hierarchizer

## Un prototype de pipeline pour la classification hiérarchique semi-automatique de concepts de thésaurus

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Ce projet est un prototype de pipeline de traitement pour la hiérarchisation semi-automatique d'un thésaurus. Développé en Python, il est conçu pour analyser des données SKOS (provenant de fichiers TSV) et enrichir leurs relations hiérarchiques de manière intelligente.

Le pipeline combine plusieurs méthodes, allant de la détection de patterns lexicaux à l'analyse sémantique, pour identifier et proposer de nouvelles relations skos:broader et skos:narrower.

## Vue d'ensemble des phases

Le pipeline est divisé en sept phases distinctes, chacune gérée par un module spécifique :

**phase1_data_processor.py:** Prétraitement et nettoyage des données brutes du thésaurus. Cette phase normalise les labels, gère les URIs et extrait les relations existantes pour les préparer à l'analyse.

**phase2_pattern_detector.py:** Détection de relations basées sur des patterns lexicaux (ex. : "flûte à bec" -> "flûte").

**phase3_similarity_analyzer.py:** Analyse de la similarité lexicale pour trouver des relations entre des termes proches (ex. : via le coefficient de Jaccard).

**phase4_contextual_embedding_analyzer.py:** Utilisation de modèles d'embeddings pour découvrir des relations par similarité sémantique, en s'appuyant sur des analogies existantes.

**phase5_hierarchy_builder.py:** Construction de la hiérarchie finale en consolidant toutes les relations proposées, en résolvant les conflits et en gérant la validation semi-automatique.

**phase6_hierarchy_optimizer.py:** Optimisation de la hiérarchie en détectant et en éliminant les redondances transitives et les cycles.

**phase7_output_generator.py:** Génération des fichiers de sortie finaux (TSV enrichi, XML RDF/SKOS) et d'un rapport de synthèse.

## Installation

Pour faire fonctionner ce projet, suivez ces étapes :

Clonez le dépôt :

```bash
# Cloner le dépôt
git clone https://github.com/AmelieDogan/thesaurus-hierarchizer.git
cd thesaurus-hierarchizer

# Créer et activer un environnement virtuel (recommandé)
python -m venv env
source env/bin/activate  # Sur Linux/Mac
# ou
env\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Exécution standard

Lancez le pipeline en spécifiant le chemin de votre fichier d'entrée TSV.

```bash
python main.py --input chemin/vers/votre_fichier.tsv
```

Option de la ligne de commande :

```--input``` (obligatoire) : Chemin du fichier TSV contenant le thésaurus.

Exemples d'utilisation avec un fichier spécifique :

```bash
python main.py --input data/thesaurus.tsv 
```

## Fichiers de sortie

Une fois le pipeline terminé, les fichiers suivants sont générés dans le dossier de sortie configuré :

```thesaurus_enriched.tsv``` : Le thésaurus original enrichi avec les nouvelles relations.

```thesaurus_hierarchy.xml``` : La hiérarchie finale au format SKOS RDF/XML.

```rapport_pipeline.html``` : Un rapport détaillé présentant les statistiques, les relations découvertes et la qualité du traitement.

Ce rapport est un excellent outil pour comprendre les résultats et évaluer la performance du pipeline sur vos données.

## Contribution

Les contributions, suggestions et retours sont les bienvenus ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request pour améliorer ce projet.