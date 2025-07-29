"""
Database module for SPARCS pair approach.

This module provides SQLite database functionality for storing and querying
star catalog data and star pairs.
"""

from .setup_db import create_database, reset_database
from .db_utils import DatabaseManager, get_db_manager, insert_star_data, get_stars_dataframe
from .load_hipparcos_data import load_hipparcos_to_database

__all__ = [
    'create_database',
    'reset_database', 
    'DatabaseManager',
    'get_db_manager',
    'insert_star_data',
    'get_stars_dataframe',
    'load_hipparcos_to_database'
] 