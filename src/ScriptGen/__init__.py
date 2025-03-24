"""
ScriptGen - A script generation module for generating scenes and dialogue.

This package provides modules for script generation from log lines:

1. Bible Storage: System to create, store, update, and retrieve "bibles" 
   (Character, Plot, Setting, Theme) as structured data.

2. LLM API Wrapper: A standardized interface to interact with various LLMs
   (OpenAI, Grok, llama, mistral, deepseek, qwen), handling prompt construction,
   response parsing, and call management.

3. Basic Script Generator: A minimal script generation engine that produces
   a single scene (outline + dialogue) from a log line, using the Bible
   Storage and LLM Wrapper.

4. Agent Framework: Implements the Director, Character, and Dramaturge agents
   as role-based prompt generators, producing and refining script components
   collaboratively.

5. Rewrite Controller: Provides scene critique and rewrite functionality with
   bible change propagation across dependent scenes.

6. Workflow Orchestrator: Integrates all components into a unified workflow for
   end-to-end script generation with optimization and parallel processing.
"""

# Import core components for easy access
from .bibles.bible_storage import BibleStorage
from .llm.llm_wrapper import LLMWrapper
from .generators.script_generator import ScriptGenerator, create_script_generator
from .agents.agent_framework import (
    CharacterAgent, 
    DirectorAgent, 
    DramaturgeAgent, 
    run_agents, 
    batch_character_agent_call
)
from .rewrite.rewrite_controller import RewriteController
from .orchestrator.workflow_controller import (
    WorkflowController,
    GenerationConfig,
    GenerationScope
)
from .orchestrator.optimizer import LLMOptimizer

# Import models
from .core.models import (
    Bible, CharacterBible, PlotBible, SettingBible, ThemeBible,
    get_engine, initialize_database
)

__all__ = [
    # Bible Storage
    'BibleStorage',
    # LLM Wrapper
    'LLMWrapper',
    # Script Generator
    'ScriptGenerator',
    'create_script_generator',
    # Database Models
    'Bible',
    'CharacterBible',
    'PlotBible',
    'SettingBible',
    'ThemeBible',
    'get_engine',
    'initialize_database',
    # Agent framework
    'CharacterAgent',
    'DirectorAgent',
    'DramaturgeAgent',
    'run_agents',
    'batch_character_agent_call',
    # Rewrite Controller
    'RewriteController',
    # Workflow Orchestrator
    'WorkflowController',
    'GenerationConfig',
    'GenerationScope',
    # Optimizer
    'LLMOptimizer'
]

# Version
__version__ = '0.3.0'
