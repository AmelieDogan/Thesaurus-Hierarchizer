"""
Fonctions utilitaires pour la normalisation de texte et la validation de mots
"""

import pandas as pd
import unicodedata
import re
import spacy
from collections import defaultdict
from typing import List, Dict, Set

from .config import get_default_stop_words, get_word_min_length

# Téléchargement du modèle français si nécessaire.
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("Modèle SpaCy 'fr_core_news_sm' non trouvé. Veuillez l'installer avec :")
    print("python -m spacy download fr_core_news_sm")
    exit()

def normalize_text(text: str) -> str:
    """
    Normalise un texte en supprimant les accents et en convertissant en minuscules.
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Le texte normalisé sans accent et sans majuscule
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Suppression des accents
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux multiples et normalisation des espaces
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_text_for_grouping(text: str) -> str:
    """
    Normalise un texte pour le regroupement en le passant en minuscule, en retirant 
    les accents et en le lemmatisant.
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Le texte lemmatisé, sans accent et en minuscules
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Étape 1: Normalisation de base (minuscule, suppression des accents)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Étape 2: Lemmatisation
    # Utilisation de SpaCy pour la lemmatisation
    doc = nlp(text.lower())  # Le modèle SpaCy gère la casse et les lemmes
    lemmatized_words = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized_words)

def is_valid_word(word: str) -> bool:
    """
    Vérifie si un mot est valide pour la détection de patterns.
    
    Args:
        word: Mot à vérifier
        
    Returns:
        True si le mot est valide (pas un mot-outil, longueur suffisante)
    """
    stop_words = get_default_stop_words()
    min_word_length = get_word_min_length()
    
    word_clean = word.strip().lower()
    return (
        len(word_clean) >= min_word_length and
        word_clean not in stop_words and
        word_clean.isalpha()  # Uniquement des lettres
    )

def group_candidate_parents(parent_list: List[str]) -> Dict[str, Set[str]]:
    """
    Regroupe les termes de parents candidats en fonction de leur forme normalisée.
    
    Args:
        parent_list (List[str]): Liste des termes parents uniques.
    
    Returns:
        Dict[str, Set[str]]: Un dictionnaire où les clés sont les termes normalisés et
                            les valeurs sont un ensemble des termes originaux correspondants.
    """
    grouped_terms = defaultdict(set)
    for term in parent_list:
        normalized_term = normalize_text_for_grouping(term)
        grouped_terms[normalized_term].add(term)
        
    return grouped_terms