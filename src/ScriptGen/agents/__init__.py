"""
Agents module for role-based prompt generators that produce and refine script components.

This module implements the Director, Character, and Dramaturge agents that work collaboratively
to generate and refine script components based on bible entries and scene contexts.
"""

from .agent_framework import (
    CharacterAgent,
    DirectorAgent,
    DramaturgeAgent,
    run_agents
)

__all__ = [
    'CharacterAgent',
    'DirectorAgent', 
    'DramaturgeAgent',
    'run_agents'
] 