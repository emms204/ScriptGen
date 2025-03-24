# ScriptGen

A modular framework for automated script generation using Large Language Models (LLMs).

## Overview

ScriptGen is a Python package that provides utilities for generating, managing, and refining script content 
using LLMs. The package is designed to be modular and extensible, with components for:

1. **Core Models and Database Abstractions**: Foundational models and database interfaces.
2. **Bible Storage**: Management of character, plot, and setting bible entries.
3. **LLM Wrapper**: Interface for LLM interaction, supporting multiple LLM providers.
4. **Script Generator**: Functions for generating script content based on bibles and prompts.
5. **Agent Framework**: Role-based agents that collaborate to produce and refine script components.

## Directory Structure

```
ScriptGen/
├── __init__.py              # Package initialization
├── core/                    # Core models and database abstractions
│   ├── __init__.py
│   └── models.py            # Database models and utility classes
├── bibles/                  # Bible storage module
│   ├── __init__.py
│   └── bible_storage.py     # Bible storage implementation
├── llm/                     # LLM wrapper module
│   ├── __init__.py
│   └── llm_wrapper.py       # LLM service wrapper
├── generators/              # Script generation modules
│   ├── __init__.py
│   └── script_generator.py  # Script generator implementation
├── agents/                  # Agent framework
│   ├── __init__.py
│   └── agent_framework.py   # Role-based agent implementation
├── utils/                   # Utility functions
│   └── __init__.py
├── config/                  # Configuration files
│   ├── models.json          # LLM configuration
│   └── prompts.json         # Prompt templates
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_bible_storage.py
│   ├── test_llm_wrapper.py
│   └── test_agent_framework.py
└── examples/                # Example scripts
    ├── __init__.py
    ├── example.py
    └── agent_example.py
```

## Components

### Core

Contains base models and database abstractions that are used throughout the package.

### Bible Storage

Manages bible entries for characters, plots, settings, etc. These entries serve as context for script generation.

### LLM Wrapper

Provides a unified interface for interacting with different LLM providers (OpenAI, etc.).

### Script Generator

Contains the core logic for generating script content based on bibles and prompts.

### Agent Framework

Implements role-based agents (Director, Character, Dramaturge) that act as collaborative prompt generators:

- **Character Agent**: Generates dialogue and actions for a specific character
- **Director Agent**: Integrates character dialogue into cohesive scenes
- **Dramaturge Agent**: Critiques scenes for pacing, tension, and subtext
- **run_agents**: Orchestrates agent calls, coordinating parallel execution

## Usage

See the `examples/` directory for usage examples:

- `example.py`: Basic usage of bible storage and script generation
- `agent_example.py`: Advanced usage with the agent framework

## Configuration

The package uses JSON configuration files located in the `config/` directory:

- `models.json`: Configuration for LLM providers
- `prompts.json`: Templates for prompts used by the script generator

## Requirements

- Python 3.8+
- SQLAlchemy
- Requests
- OpenAI (for API access)

## Development

1. Clone the repository
2. Install the requirements: `pip install -r requirements.txt`
3. Run the tests: `python -m unittest discover src/TextGen/ScriptGen/tests` 