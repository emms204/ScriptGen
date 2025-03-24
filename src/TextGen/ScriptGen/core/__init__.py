"""
Core module for common functionality and base classes.
"""

from .models import (
    Bible, CharacterBible, PlotBible, SettingBible, ThemeBible,
    get_engine, initialize_database
)

__all__ = [
    'Bible', 
    'CharacterBible', 
    'PlotBible', 
    'SettingBible', 
    'ThemeBible',
    'get_engine', 
    'initialize_database'
]
