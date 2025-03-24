"""
Orchestrator module for workflow management and script generation pipeline.

This module provides the central controller components that orchestrate the full
script generation process:

1. WorkflowController: Manages the end-to-end script generation pipeline
2. ScriptProcessor: Handles multi-scene script generation and scene connections
3. LLMOptimizer: Optimizes LLM usage through caching and batching
"""

from .workflow_controller import (
    WorkflowController, 
    ScriptProcessor,
    GenerationScope,
    GenerationConfig,
    WorkflowStep
)

from .optimizer import (
    LLMOptimizer,
    CacheConfig,
    OptimizationStats
)

__all__ = [
    'WorkflowController',
    'ScriptProcessor',
    'GenerationScope',
    'GenerationConfig',
    'WorkflowStep',
    'LLMOptimizer',
    'CacheConfig',
    'OptimizationStats'
] 