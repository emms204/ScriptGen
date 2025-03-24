# ScriptGen Project Structure

This document outlines the organization of the ScriptGen project and provides guidance for future development phases.

## Current Directory Structure

```
src/TextGen/ScriptGen/
├── __init__.py                # Package initialization and exports
├── README.md                  # Project documentation
├── IMPLEMENTATION_SUMMARY.md  # Summary of implemented features
├── STRUCTURE.md               # This file - project structure guide
├── bibles/                    # Bible storage and management
│   ├── __init__.py
│   └── bible_storage.py
├── config/                    # Configuration files
│   ├── models.json            # LLM model configurations
│   └── prompts.json           # Prompt templates
├── core/                      # Core components and models
│   ├── __init__.py
│   └── models.py              # Database models
├── examples/                  # Example scripts and demos
│   ├── __init__.py
│   └── example.py             # Basic script generation example
├── generators/                # Script generation engines
│   ├── __init__.py
│   └── script_generator.py    # Basic script generator
├── llm/                       # LLM API wrappers
│   ├── __init__.py
│   └── llm_wrapper.py         # LLM standardized interface
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   ├── test_bible_storage.py  # Bible storage tests
│   └── test_llm_wrapper.py    # LLM wrapper tests
└── utils/                     # Utility functions and helpers
    └── __init__.py
```

## Phase 1: Core Infrastructure (Completed)

- ✅ Bible Storage and Management: Centralized system for character, plot, setting, and theme data
- ✅ LLM API Wrapper: Standardized interface for multiple LLM providers
- ✅ Basic Script Generator: Scene generation from log lines

## Phase 2: Advanced Generation and Consistency

### Implementation Plan

- **Character Development Module** (in `generators/character_generator.py`)
  - Character arc generation
  - Personality trait consistency
  - Background and motivation development

- **Dialogue Enhancement** (in `generators/dialogue_generator.py`)
  - Character voice consistency
  - Multi-character conversation flow
  - Subtext and emotional layer generation

- **Plot Structuring** (in `generators/plot_generator.py`)
  - Multi-scene script generation
  - Plot arc development
  - Scene sequencing and transitions

## Phase 3: Memory and Context

### Implementation Plan

- **Long-term Memory** (in `bibles/memory_storage.py`)
  - Cross-scene/script consistency tracking
  - Character relationship evolution
  - Plot thread management

- **Contextual Awareness** (in `utils/context_manager.py`)
  - World state tracking
  - Causal chain maintenance
  - Event consequence modeling

## Phase 4: Refinement and Editing

### Implementation Plan

- **Script Editor** (in `generators/script_editor.py`)
  - Dialogue polishing
  - Scene tightening
  - Narrative flow improvements

- **Style Adaptation** (in `generators/style_adapter.py`)
  - Genre-specific adaptations
  - Tone consistency
  - Voice and style control

## Phase 5: Interactive Generation

### Implementation Plan

- **User Feedback Integration** (in `utils/feedback_manager.py`)
  - Change request processing
  - Iterative improvement loop
  - Preference learning

- **Collaboration Interface** (in `utils/collaboration_tools.py`)
  - Suggestion generation
  - Alternative scene proposals
  - Explanation capabilities

## Phase 6: Advanced Applications

### Implementation Plan

- **Multi-format Output** (in `generators/format_converter.py`)
  - Screenplay standard formatting
  - Novel/prose adaptation
  - Comic/graphic novel paneling

- **Character Graph Visualization** (in `utils/visualizers.py`)
  - Relationship network graphs
  - Character arc visualization
  - Plot structure diagrams

## Integration Guidelines

When adding new modules to the system:

1. **Maintain dependency hierarchy**:
   - Core modules should have minimal dependencies
   - Higher-level modules can depend on lower-level ones
   - Avoid circular dependencies

2. **File organization**:
   - Place modules in the appropriate subdirectory
   - Create new subdirectories if needed for major feature sets
   - Update `__init__.py` files to expose necessary components

3. **Testing**:
   - Add unit tests for all new modules in the `tests/` directory
   - Name test files with the `test_` prefix
   - Update example scripts to demonstrate new functionality

4. **Documentation**:
   - Document classes and functions with docstrings
   - Update README.md with new features
   - Add examples demonstrating new functionality

## Style Guidelines

- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Document all public functions and classes
- Use meaningful variable and function names
- Keep functions focused on a single responsibility 