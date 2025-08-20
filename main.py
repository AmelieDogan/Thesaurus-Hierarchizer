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
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Chemin vers un fichier de configuration JSON personnalisé (facultatif)."
    )
    
    parser.add_argument(
        "--sample-validation",
        action="store_true",
        help="Si activé, génère un échantillon de relations pour validation manuelle."
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Taille de l’échantillon à générer pour validation (avec --sample-validation)."
    )
    
    return parser.parse_args()

def main():
    """
    Point d’entrée principal du script.
    """
    args = parse_arguments()
    
    # Charger configuration personnalisée si fournie
    config = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    # Initialisation de l’orchestrateur
    builder = ThesaurusHierarchyBuilder(tsv_file_path=args.input, config=config)
    
    # Exécution complète du pipeline
    builder.run_pipeline()

if __name__ == "__main__":
    main()
