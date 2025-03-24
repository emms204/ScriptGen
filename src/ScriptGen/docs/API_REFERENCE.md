# ScriptGen API Reference

This document provides a comprehensive reference for the ScriptGen API, including classes, methods, and usage patterns.

## Table of Contents

1. [Core API](#core-api)
   - [ScriptGenAPI](#scriptgenapi)
   - [WorkflowController](#workflowcontroller)
   - [LLMOptimizer](#llmoptimizer)
2. [Bible Management API](#bible-management-api)
   - [BibleStorage](#biblestorage)
   - [BibleEntry](#bibleentry)
3. [Generation API](#generation-api)
   - [ScriptGenerator](#scriptgenerator)
   - [GenerationConfig](#generationconfig)
4. [Agent API](#agent-api)
   - [AgentFramework](#agentframework)
   - [CharacterAgent](#characteragent)
   - [DirectorAgent](#directoragent)
   - [DramaturgeAgent](#dramaturgeagent)
5. [Rewrite API](#rewrite-api)
   - [RewriteController](#rewritecontroller)
6. [Utility API](#utility-api)
   - [Config Management](#config-management)
   - [Logging](#logging)
7. [Error Handling](#error-handling)
   - [Error Types](#error-types)
   - [Error Strategies](#error-strategies)

## Core API

### ScriptGenAPI

The main API class that serves as the primary interface for client applications.

```python
class ScriptGenAPI:
    def __init__(
        self, 
        config_path=None,
        llm_provider="openai",
        use_optimizer=True
    ):
        """
        Initialize the ScriptGen API.
        
        Args:
            config_path (str, optional): Path to configuration file
            llm_provider (str, optional): LLM provider to use ("openai", "anthropic", etc.)
            use_optimizer (bool, optional): Whether to use LLM optimization
        """
    
    def generate_scene(
        self,
        log_line,
        character_count=2,
        llm_settings=None,
        existing_bibles=None,
        auto_rewrite=False
    ):
        """
        Generate a single scene from a log line.
        
        Args:
            log_line (str): The log line describing the scene
            character_count (int, optional): Number of characters to generate
            llm_settings (dict, optional): Custom LLM settings
            existing_bibles (dict, optional): Dictionary of existing bible entries
            auto_rewrite (bool, optional): Whether to apply automatic rewrites
            
        Returns:
            dict: The generated scene data
        """
    
    def generate_script(
        self,
        log_line,
        scene_count=3,
        character_count=3,
        llm_settings=None,
        existing_bibles=None,
        auto_rewrite=False,
        parallel_generation=True
    ):
        """
        Generate a multi-scene script from a log line.
        
        Args:
            log_line (str): The log line describing the story
            scene_count (int, optional): Number of scenes to generate
            character_count (int, optional): Number of characters to generate
            llm_settings (dict, optional): Custom LLM settings
            existing_bibles (dict, optional): Dictionary of existing bible entries
            auto_rewrite (bool, optional): Whether to apply automatic rewrites
            parallel_generation (bool, optional): Whether to generate scenes in parallel
            
        Returns:
            dict: The generated script data
        """
    
    def rewrite_scene(
        self,
        scene_data,
        critique=None,
        bible_updates=None
    ):
        """
        Rewrite a scene based on critique or bible updates.
        
        Args:
            scene_data (dict): The scene data to rewrite
            critique (str, optional): Critique to address in the rewrite
            bible_updates (dict, optional): Updates to bible entries
            
        Returns:
            dict: The rewritten scene data
        """
    
    def create_bible_entry(
        self,
        entry_type,
        data
    ):
        """
        Create a new bible entry.
        
        Args:
            entry_type (str): Type of bible entry ("character", "plot", "setting", "theme")
            data (dict): Data for the bible entry
            
        Returns:
            str: ID of the created bible entry
        """
    
    def get_bible_entry(
        self,
        entry_id
    ):
        """
        Get a bible entry by ID.
        
        Args:
            entry_id (str): ID of the bible entry
            
        Returns:
            dict: The bible entry data
        """
    
    def update_bible_entry(
        self,
        entry_id,
        data
    ):
        """
        Update a bible entry.
        
        Args:
            entry_id (str): ID of the bible entry
            data (dict): Updated data for the bible entry
            
        Returns:
            dict: The updated bible entry data
        """
    
    def get_optimization_stats(self):
        """
        Get statistics from the LLM optimizer.
        
        Returns:
            dict: Optimization statistics
        """
```

### WorkflowController

The central controller class that orchestrates the script generation pipeline.

```python
class WorkflowController:
    def __init__(
        self,
        bible_storage=None,
        llm_wrapper=None,
        script_generator=None,
        rewrite_controller=None,
        llm_optimizer=None,
        config_path=None
    ):
        """
        Initialize the workflow controller.
        
        Args:
            bible_storage (BibleStorage, optional): Bible storage instance
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
            script_generator (ScriptGenerator, optional): Script generator instance
            rewrite_controller (RewriteController, optional): Rewrite controller instance
            llm_optimizer (LLMOptimizer, optional): LLM optimizer instance
            config_path (str, optional): Path to configuration file
        """
    
    def generate_script(
        self,
        log_line,
        config=None,
        existing_bibles=None
    ):
        """
        Generate a script based on a log line.
        
        Args:
            log_line (str): The log line describing the story
            config (GenerationConfig, optional): Configuration for generation
            existing_bibles (dict, optional): Dictionary of existing bible entries
            
        Returns:
            dict: The generated script data
        """
    
    def bootstrap_bibles(
        self,
        log_line,
        character_count=None,
        existing_bibles=None
    ):
        """
        Bootstrap bible entries from a log line.
        
        Args:
            log_line (str): The log line describing the story
            character_count (int, optional): Number of characters to generate
            existing_bibles (dict, optional): Dictionary of existing bible entries
            
        Returns:
            dict: The generated bible entries
        """
    
    def generate_scene_outline(
        self,
        log_line,
        bibles,
        scene_number=0
    ):
        """
        Generate an outline for a scene.
        
        Args:
            log_line (str): The log line describing the story
            bibles (dict): Bible entries to use for generation
            scene_number (int, optional): Scene number in the sequence
            
        Returns:
            dict: The generated scene outline
        """
    
    def generate_scene(
        self,
        outline,
        bibles,
        config=None
    ):
        """
        Generate a scene from an outline.
        
        Args:
            outline (dict): Scene outline
            bibles (dict): Bible entries to use for generation
            config (GenerationConfig, optional): Configuration for generation
            
        Returns:
            dict: The generated scene data
        """
    
    def rewrite_scene(
        self,
        scene,
        bibles,
        critique=None,
        bible_updates=None
    ):
        """
        Rewrite a scene based on critique or bible updates.
        
        Args:
            scene (dict): The scene data to rewrite
            bibles (dict): Bible entries to use for generation
            critique (str, optional): Critique to address in the rewrite
            bible_updates (dict, optional): Updates to bible entries
            
        Returns:
            dict: The rewritten scene data
        """
    
    def format_script(
        self,
        scenes,
        bibles
    ):
        """
        Format scenes into a complete script.
        
        Args:
            scenes (list): List of generated scenes
            bibles (dict): Bible entries used for generation
            
        Returns:
            dict: The formatted script data
        """
```

### GenerationConfig

Configuration class for script generation.

```python
class GenerationConfig:
    def __init__(
        self,
        scope=GenerationScope.SINGLE_SCENE,
        character_count=2,
        scene_count=1,
        use_existing_bibles=False,
        auto_rewrite=False,
        rewrite_iterations=1,
        parallel_generation=True,
        optimize_llm_calls=True,
        cache_config=None
    ):
        """
        Configure script generation parameters.
        
        Args:
            scope (GenerationScope): Scope of generation (SINGLE_SCENE, MULTI_SCENE, FULL_SCRIPT)
            character_count (int): Number of characters to generate
            scene_count (int): Number of scenes to generate
            use_existing_bibles (bool): Whether to use existing bibles
            auto_rewrite (bool): Whether to apply automatic rewrites
            rewrite_iterations (int): Number of rewrite iterations to perform
            parallel_generation (bool): Whether to generate scenes in parallel
            optimize_llm_calls (bool): Whether to use LLM optimization
            cache_config (dict): Configuration for response caching
        """
```

### GenerationScope

Enum for generation scope.

```python
class GenerationScope(Enum):
    SINGLE_SCENE = "single_scene"  # Generate a single scene
    MULTI_SCENE = "multi_scene"    # Generate multiple connected scenes
    FULL_SCRIPT = "full_script"    # Generate a complete script with act structure
```

### LLMOptimizer

Optimization class for LLM calls.

```python
class LLMOptimizer:
    def __init__(
        self,
        llm_wrapper=None,
        config=None
    ):
        """
        Initialize LLM optimizer.
        
        Args:
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
            config (OptimizerConfig, optional): Configuration for optimization
        """
    
    def call_llm(
        self,
        prompt,
        model=None,
        params=None
    ):
        """
        Call LLM with optimization.
        
        Args:
            prompt (str): The prompt to send to the LLM
            model (str, optional): The model to use
            params (dict, optional): Additional parameters for the call
            
        Returns:
            str: The LLM response
        """
    
    def start_batch(self):
        """
        Start batch mode for collecting multiple LLM calls.
        """
    
    def end_batch(self):
        """
        End batch mode and process all queued calls.
        
        Returns:
            list: List of responses for all batched calls
        """
    
    def get_stats(self):
        """
        Get current optimization statistics.
        
        Returns:
            dict: Optimization statistics
        """
    
    def clear_cache(self):
        """
        Clear the response cache.
        """
```

### OptimizerConfig

Configuration class for LLM optimization.

```python
class OptimizerConfig:
    def __init__(
        self,
        enable_caching=True,
        cache_expire_hours=24,
        enable_batching=True,
        batch_timeout_ms=500,
        enable_deduplication=True,
        dedup_window_seconds=30
    ):
        """
        Configure LLM optimization parameters.
        
        Args:
            enable_caching (bool): Whether to enable response caching
            cache_expire_hours (int): Cache expiration time in hours
            enable_batching (bool): Whether to enable call batching
            batch_timeout_ms (int): Batch collection timeout in milliseconds
            enable_deduplication (bool): Whether to enable prompt deduplication
            dedup_window_seconds (int): Deduplication time window in seconds
        """
```

## Bible Management API

### BibleStorage

Class for storing and retrieving bible entries.

```python
class BibleStorage:
    def __init__(
        self,
        db_path=None
    ):
        """
        Initialize bible storage.
        
        Args:
            db_path (str, optional): Path to SQLite database file
        """
    
    def create_entry(
        self,
        entry_type,
        data
    ):
        """
        Create a new bible entry.
        
        Args:
            entry_type (str): Type of bible entry ("character", "plot", "setting", "theme")
            data (dict): Data for the bible entry
            
        Returns:
            str: ID of the created bible entry
        """
    
    def get_entry(
        self,
        entry_id
    ):
        """
        Get a bible entry by ID.
        
        Args:
            entry_id (str): ID of the bible entry
            
        Returns:
            BibleEntry: The bible entry
        """
    
    def update_entry(
        self,
        entry_id,
        data
    ):
        """
        Update a bible entry.
        
        Args:
            entry_id (str): ID of the bible entry
            data (dict): Updated data for the bible entry
            
        Returns:
            BibleEntry: The updated bible entry
        """
    
    def delete_entry(
        self,
        entry_id
    ):
        """
        Delete a bible entry.
        
        Args:
            entry_id (str): ID of the bible entry
            
        Returns:
            bool: Success status
        """
    
    def list_entries(
        self,
        entry_type=None
    ):
        """
        List bible entries.
        
        Args:
            entry_type (str, optional): Type of bible entries to list
            
        Returns:
            list: List of bible entries
        """
```

### BibleEntry

Class representing a bible entry.

```python
class BibleEntry:
    """
    A bible entry with type, data, and versioning.
    
    Attributes:
        id (str): Unique identifier
        type (str): Entry type ("character", "plot", "setting", "theme")
        data (dict): Entry data
        created_at (datetime): Creation timestamp
        updated_at (datetime): Last update timestamp
        version (int): Entry version
    """
```

## Generation API

### ScriptGenerator

Class for generating script components.

```python
class ScriptGenerator:
    def __init__(
        self,
        llm_wrapper=None,
        config_path=None
    ):
        """
        Initialize script generator.
        
        Args:
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
            config_path (str, optional): Path to configuration file
        """
    
    def generate_scene_outline(
        self,
        log_line,
        bibles,
        scene_number=0
    ):
        """
        Generate an outline for a scene.
        
        Args:
            log_line (str): The log line describing the story
            bibles (dict): Bible entries to use for generation
            scene_number (int, optional): Scene number in the sequence
            
        Returns:
            dict: The generated scene outline
        """
    
    def generate_character_dialogues(
        self,
        outline,
        bibles,
        agents=None
    ):
        """
        Generate character dialogues for a scene.
        
        Args:
            outline (dict): Scene outline
            bibles (dict): Bible entries to use for generation
            agents (dict, optional): Agent instances to use
            
        Returns:
            dict: The generated character dialogues
        """
    
    def generate_integrated_scene(
        self,
        outline,
        dialogues,
        bibles,
        agent=None
    ):
        """
        Generate an integrated scene from outline and dialogues.
        
        Args:
            outline (dict): Scene outline
            dialogues (dict): Character dialogues
            bibles (dict): Bible entries to use for generation
            agent (DirectorAgent, optional): Director agent instance
            
        Returns:
            str: The integrated scene
        """
    
    def format_screenplay(
        self,
        integrated_scene
    ):
        """
        Format an integrated scene as a screenplay.
        
        Args:
            integrated_scene (str): The integrated scene
            
        Returns:
            str: The formatted screenplay
        """
```

## Agent API

### AgentFramework

Functions for working with agents.

```python
def run_agents(
    outline,
    bibles,
    character_agents=None,
    director_agent=None,
    dramaturge_agent=None,
    batch_character_agents=False
):
    """
    Run agents to generate a scene.
    
    Args:
        outline (dict): Scene outline
        bibles (dict): Bible entries to use for generation
        character_agents (dict, optional): Character agent instances
        director_agent (DirectorAgent, optional): Director agent instance
        dramaturge_agent (DramaturgeAgent, optional): Dramaturge agent instance
        batch_character_agents (bool, optional): Whether to batch character agent calls
        
    Returns:
        dict: The generated scene data
    """
```

### CharacterAgent

Agent for generating character dialogue.

```python
class CharacterAgent:
    def __init__(
        self,
        llm_wrapper=None,
        character_bible=None
    ):
        """
        Initialize character agent.
        
        Args:
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
            character_bible (dict, optional): Character bible data
        """
    
    def generate_dialogue(
        self,
        outline,
        bibles,
        format_output=True
    ):
        """
        Generate dialogue for a character.
        
        Args:
            outline (dict): Scene outline
            bibles (dict): Bible entries to use for generation
            format_output (bool, optional): Whether to format the output
            
        Returns:
            str: The generated dialogue
        """
```

### DirectorAgent

Agent for integrating character dialogue into a scene.

```python
class DirectorAgent:
    def __init__(
        self,
        llm_wrapper=None
    ):
        """
        Initialize director agent.
        
        Args:
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
        """
    
    def integrate_scene(
        self,
        outline,
        dialogues,
        bibles
    ):
        """
        Integrate character dialogues into a scene.
        
        Args:
            outline (dict): Scene outline
            dialogues (dict): Character dialogues
            bibles (dict): Bible entries to use for generation
            
        Returns:
            str: The integrated scene
        """
```

### DramaturgeAgent

Agent for critiquing and suggesting rewrites.

```python
class DramaturgeAgent:
    def __init__(
        self,
        llm_wrapper=None
    ):
        """
        Initialize dramaturge agent.
        
        Args:
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
        """
    
    def critique_scene(
        self,
        scene,
        bibles
    ):
        """
        Critique a scene.
        
        Args:
            scene (str): The scene to critique
            bibles (dict): Bible entries to use for generation
            
        Returns:
            dict: The critique data
        """
    
    def suggest_rewrites(
        self,
        scene,
        critique,
        bibles
    ):
        """
        Suggest rewrites for a scene.
        
        Args:
            scene (str): The scene to rewrite
            critique (dict): Critique data
            bibles (dict): Bible entries to use for generation
            
        Returns:
            dict: The rewrite suggestions
        """
```

## Rewrite API

### RewriteController

Class for rewriting scenes.

```python
class RewriteController:
    def __init__(
        self,
        llm_wrapper=None,
        dramaturge_agent=None
    ):
        """
        Initialize rewrite controller.
        
        Args:
            llm_wrapper (LLMWrapper, optional): LLM wrapper instance
            dramaturge_agent (DramaturgeAgent, optional): Dramaturge agent instance
        """
    
    def rewrite_scene(
        self,
        scene,
        bibles,
        critique=None,
        iterations=1
    ):
        """
        Rewrite a scene.
        
        Args:
            scene (dict): The scene data to rewrite
            bibles (dict): Bible entries to use for generation
            critique (str, optional): Critique to address in the rewrite
            iterations (int, optional): Number of rewrite iterations
            
        Returns:
            dict: The rewritten scene data
        """
    
    def apply_bible_updates(
        self,
        scene,
        bibles,
        bible_updates
    ):
        """
        Apply bible updates to a scene.
        
        Args:
            scene (dict): The scene data to update
            bibles (dict): Original bible entries
            bible_updates (dict): Updates to bible entries
            
        Returns:
            dict: The updated scene data
        """
```

## Utility API

### Config Management

Functions for managing configuration.

```python
def load_config(config_path):
    """
    Load configuration from a file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: The loaded configuration
    """

def merge_configs(base_config, override_config):
    """
    Merge two configurations.
    
    Args:
        base_config (dict): Base configuration
        override_config (dict): Override configuration
        
    Returns:
        dict: The merged configuration
    """
```

### Logging

Functions for logging.

```python
def setup_logging(config=None):
    """
    Set up logging.
    
    Args:
        config (dict, optional): Logging configuration
    """

def get_logger(name):
    """
    Get a logger.
    
    Args:
        name (str): Logger name
        
    Returns:
        Logger: The logger
    """
```

## Error Handling

### Error Types

```python
class BibleError(Exception):
    """Error related to bible operations."""
    
class LLMError(Exception):
    """Error related to LLM operations."""
    
class GenerationError(Exception):
    """Error related to script generation."""
    
class AgentError(Exception):
    """Error related to agent operations."""
    
class RewriteError(Exception):
    """Error related to rewrite operations."""
    
class OptimizationError(Exception):
    """Error related to optimization operations."""
```

### Error Strategies

Available error handling strategies:

- `retry`: Retry the operation
- `fallback`: Use a fallback implementation
- `create_default`: Create default data and continue
- `propagate`: Propagate the error to the caller
- `log_and_continue`: Log the error and continue with partial data

## Example Usage

### Basic Script Generation

```python
from ScriptGen.frontend.utils.api import ScriptGenAPI

# Initialize API
api = ScriptGenAPI(use_optimizer=True)

# Generate a scene
scene = api.generate_scene(
    log_line="A detective discovers a crucial clue in an unexpected place.",
    character_count=3,
    auto_rewrite=True
)

# Print the generated scene
print(scene['integrated_scene'])
```

### Multi-Scene Script Generation

```python
# Generate a multi-scene script
script = api.generate_script(
    log_line="A family vacation turns into a survival adventure when their plane crashes.",
    scene_count=3,
    character_count=4,
    parallel_generation=True
)

# Process scenes
for i, scene in enumerate(script['scenes']):
    print(f"Scene {i+1}: {scene['outline']['description']}")
    print(scene['integrated_scene'])
    print("\n---\n")
```

### Bible Management and Rewriting

```python
# Create character bibles
detective_id = api.create_bible_entry('character', {
    'name': 'Detective Smith',
    'backstory': 'A veteran detective with amnesia',
    'traits': ['determined', 'confused', 'intelligent']
})

witness_id = api.create_bible_entry('character', {
    'name': 'Dr. Wilson',
    'backstory': 'A forensic psychologist helping the detective',
    'traits': ['analytical', 'cautious', 'empathetic']
})

# Generate a scene with existing bibles
scene = api.generate_scene(
    log_line="The detective interrogates a key witness.",
    existing_bibles={
        'character': [detective_id, witness_id]
    }
)

# Rewrite the scene with a critique
rewritten_scene = api.rewrite_scene(
    scene_data=scene,
    critique="Increase the tension between the characters and add more subtext."
)

# Print the rewritten scene
print(rewritten_scene['integrated_scene'])
```

### Optimization Statistics

```python
# Generate a script with optimization
script = api.generate_script(
    log_line="A heist gone wrong leads to unexpected alliances.",
    scene_count=5,
    character_count=5
)

# Get optimization statistics
stats = api.get_optimization_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Total calls: {stats['total_calls']}")
print(f"Estimated savings: ${stats['estimated_cost_savings']:.2f}")
print(f"Tokens saved: {stats['tokens_saved']}")
``` 