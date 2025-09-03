# -*- coding: utf-8 -*-
"""
Initialise la config de logging une seule fois
"""

from .config import setup_logging
import logging

setup_logging()

def get_logger(name=None):
    return logging.getLogger(name)