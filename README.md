# Thesaurus-Hierarchizer

> Projet en cours de développement.

## Un framework pour la classification hiérarchique automatique de concepts de thésaurus spécialisés

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/CC_BY_NC_SA-4.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Ce projet propose une architecture intégrée pour structurer automatiquement des thésaurus spécialisés en français, en combinant techniques d'embeddings, analyse terminologique, et méthodes basées sur les graphes.

## Fonctionnalités

- **Génération d'embeddings** adaptés au domaine spécialisé français
- **Extraction de relations hiérarchiques** par patrons linguistiques
- **Clustering hiérarchique** avec contraintes basées sur les relations extraites
- **Analyse de graphe** pour identifier les niveaux conceptuels
- **Raffinement de la hiérarchie** pour assurer la cohérence taxonomique
- **Visualisation interactive** des hiérarchies et des relations conceptuelles
- **Évaluation multi-critères** des taxonomies générées

## Prérequis

- Python 3.8+
- Dépendances listées dans `requirements.txt`

## Installation

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

# Installation en mode développement (optionnel)
pip install -e .
```

## Structure du projet

```
thesaurus-hierarchizer/
│
├── data/                   # Données et ressources
├── config/                 # Fichiers de configuration
├── notebooks/              # Notebooks d'exploration
├── src/                    # Code source du projet
│   ├── data/               # Gestion des données
│   ├── embeddings/         # Génération d'embeddings
│   ├── extraction/         # Extraction de relations
│   ├── clustering/         # Algorithmes de clustering
│   ├── graph/              # Modélisation et analyse de graphes
│   ├── refinement/         # Raffinement de la hiérarchie
│   ├── visualization/      # Visualisation des résultats
│   ├── evaluation/         # Évaluation des résultats
│   └── pipeline/           # Orchestration du processus
│
└── tests/                  # Tests unitaires et d'intégration
```

## Architecture

Le projet suit une architecture en trois phases principales:

1. **Phase d'initialisation:**
   - Génération d'embeddings spécifiques au domaine
   - Extraction de relations candidates par patterns linguistiques

2. **Phase de structuration:**
   - Clustering hiérarchique contraint par les relations extraites
   - Construction d'un graphe de concepts avec pondération multiple

3. **Phase de raffinement:**
   - Validation par inférence logique
   - Détection et résolution des incohérences
   - Enrichissement par sources externes

## Cas d'utilisation

Ce framework est particulièrement adapté pour:
- Thésaurus spécialisés en français nécessitant une structuration hiérarchique
- Domaines techniques avec terminologie précise et définitions formelles
- Création ou refonte de taxonomies pour l'organisation de connaissances
- Enrichissement de systèmes d'information avec des relations sémantiques

## Évaluation

Le module d'évaluation permet de mesurer la qualité des hiérarchies générées selon plusieurs critères:
- Cohérence taxonomique
- Profondeur et équilibre de la hiérarchie
- Comparaison avec des taxonomies de référence
- Pertinence sémantique des relations parent-enfant

## Contribution

Les contributions sont les bienvenues ! Pour contribuer:

1. Forkez le dépôt
2. Créez une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commitez vos changements (`git commit -m 'Ajout de fonctionnalité X'`)
4. Poussez vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence CC BY-NC-SA 4.0 - voir le fichier [LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/) pour plus de détails.

## Contact

Pour toute question, suggestion ou collaboration, n'hésitez pas à ouvrir une issue ou à me contacter directement.
