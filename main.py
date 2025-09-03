import argparse
import json
from app.logger import get_logger
from app.main_orchestrator import ThesaurusHierarchyBuilder

logger = get_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        Namespace contenant les arguments parsés.
    """
    parser = argparse.ArgumentParser(description="Hiérarchisation d’un thésaurus musical.")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin du fichier TSV du thésaurus à traiter."
    )
    
    return parser.parse_args()

def main():
    """
    Point d’entrée principal du script.
    """
    args = parse_arguments()
    
    # Initialisation de l’orchestrateur
    builder = ThesaurusHierarchyBuilder(tsv_file_path=args.input)
    
    # Exécution complète du pipeline
    builder.run_pipeline()

if __name__ == "__main__":
    main()
