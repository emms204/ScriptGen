# ScriptGen Implementation Summary

## Project Overview

ScriptGen is a modular framework for automated script generation using Large Language Models (LLMs). The system is designed to produce screenplay scenes using structured data ("bibles") for characters, plots, and settings to maintain consistency and context.

## Directory Structure

The codebase is organized into the following directory structure:

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
├── rewrite/                 # Rewrite module
│   ├── __init__.py
│   └── rewrite_controller.py # Scene rewrite implementation
├── orchestrator/            # System integration and workflow orchestration
│   ├── __init__.py
│   ├── workflow_controller.py # Workflow orchestration
│   └── optimizer.py         # LLM optimization
├── utils/                   # Utility functions
│   └── __init__.py
├── config/                  # Configuration files
│   ├── models.json          # LLM configuration
│   └── prompts.json         # Prompt templates
├── docs/                    # Documentation
│   ├── SYSTEM_INTEGRATION.md # Integration documentation
│   └── API_REFERENCE.md     # API reference
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_bible_storage.py
│   ├── test_llm_wrapper.py
│   ├── test_agent_framework.py
│   └── test_workflow_controller.py
└── examples/                # Example scripts
    ├── __init__.py
    ├── example.py
    ├── agent_example.py
    └── workflow_example.py
```

## Core Modules

### 1. Bible Storage (`bibles/bible_storage.py`)

A centralized system to create, store, update, and retrieve "bibles" (Character, Plot, Setting, Theme) as structured data.

Key features:
- SQLAlchemy-based database for storing bible entries
- CRUD operations for bible entries
- Versioning of bible entries
- Type validation and schema enforcement

### 2. LLM Wrapper (`llm/llm_wrapper.py`)

A standardized interface to interact with various LLMs (OpenAI, etc.), handling prompt construction, response parsing, and call management.

Key features:
- Support for multiple LLM providers
- Configurable via JSON config
- Error handling and retries
- Response parsing

### 3. Script Generator (`generators/script_generator.py`)

A script generation engine that produces scenes from log lines or bible entries.

Key features:
- Scene outline generation
- Character dialogue generation
- Screenplay formatting
- Integration with Bible Storage and LLM Wrapper

### 4. Agent Framework (`agents/agent_framework.py`)

A framework of role-based agents that collaborate to produce and refine script components.

Key features:
- Character Agent: Generates dialogue for specific characters
- Director Agent: Integrates character dialogue into cohesive scenes
- Dramaturge Agent: Critiques scenes for pacing, tension, and subtext
- Orchestration through the `run_agents` function
- Optional batching of character agent calls

### 5. Rewrite Controller (`rewrite/rewrite_controller.py`)

A module for iterative script refinement and rewriting based on critiques.

Key features:
- Scene critique generation
- Targeted rewrite suggestions
- Bible change propagation
- Incremental improvements through multiple iterations

### 6. Workflow Orchestrator (`orchestrator/workflow_controller.py`)

A central controller for managing the entire script generation pipeline.

Key features:
- End-to-end script generation workflow
- Parallel processing for multi-scene generation
- Support for different generation scopes (single scene, multi-scene, full script)
- Integration of all system components
- Configurable generation parameters

### 7. LLM Optimizer (`orchestrator/optimizer.py`)

A module for optimizing LLM usage through various techniques.

Key features:
- Response caching for reusing common LLM responses
- Call batching for improved throughput
- Prompt deduplication for reduced API costs
- Token usage tracking and optimization
- Performance metrics and statistics

## Dependencies

- SQLAlchemy for database operations
- OpenAI API for LLM access
- Python 3.8+ required
- Threading and concurrent.futures for parallel processing

## Implementation Status

Phase 1 (Complete):
- Core models and database implementation
- Bible storage implementation
- LLM wrapper for API interaction
- Basic script generator

Phase 2 (Complete):
- Agent framework implementation
- Directory restructuring for better organization
- Creation of examples and tests
- Documentation updates

Phase 3 (Complete):
- Scene rewrite controller for iterative refinement
- Advanced dialogue formatting and screenplay conventions
- Frontend API enhancements
- Integration testing

Phase 4 (Complete):
- System integration through the workflow orchestrator
- LLM optimization for improved performance and cost reduction
- Parallel processing for multi-scene generation
- Comprehensive documentation and examples
- Frontend API integration

## Running the Examples

See the examples directory for usage:
- `example.py`: Basic usage of bible storage and script generation
- `agent_example.py`: Advanced usage with the agent framework
- `workflow_example.py`: End-to-end script generation using the workflow orchestrator

## Configuration

The `config/` directory contains configuration files:
- `models.json`: Configuration for LLM providers
- `prompts.json`: Templates for prompts used by the generators 

## Documentation

The `docs/` directory contains detailed documentation:
- `SYSTEM_INTEGRATION.md`: Comprehensive guide to system integration
- `API_REFERENCE.md`: Reference for public APIs 