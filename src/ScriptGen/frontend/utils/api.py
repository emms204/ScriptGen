"""
API client for backend communication.

This module provides a client for communicating with the ScriptGen backend,
either through direct function calls or REST API depending on configuration.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

# Import backend components for direct function calls
from ...bibles.bible_storage import BibleStorage
from ...llm.llm_wrapper import LLMWrapper
from ...generators.script_generator import ScriptGenerator
from ...rewrite.rewrite_controller import (
    RewriteController, ChangeType, ChangeRequest, SceneCritique, ChangePlan
)
from ...agents.agent_framework import run_agents
from ...orchestrator.workflow_controller import (
    WorkflowController, GenerationConfig, GenerationScope
)
from ...orchestrator.optimizer import LLMOptimizer, CacheConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scriptgen_api')


class ScriptGenAPI:
    """
    Client for ScriptGen backend communication.
    
    This class provides methods to interact with the ScriptGen backend,
    either through direct function calls (embedded mode) or REST API calls.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for REST API. If None, use direct function calls.
        """
        self.base_url = base_url
        self.direct_mode = base_url is None
        
        # Initialize components for direct function calls if in direct mode
        if self.direct_mode:
            self.bible_storage = BibleStorage()
            self.llm_wrapper = LLMWrapper()
            self.script_generator = ScriptGenerator(
                bible_storage=self.bible_storage,
                llm_wrapper=self.llm_wrapper
            )
            self.rewrite_controller = RewriteController(
                bible_storage=self.bible_storage,
                llm_wrapper=self.llm_wrapper
            )
            self.llm_optimizer = LLMOptimizer(
                llm_wrapper=self.llm_wrapper,
                config=CacheConfig(enable_cache=True)
            )
            self.workflow_controller = WorkflowController(
                bible_storage=self.bible_storage,
                llm_wrapper=self.llm_wrapper,
                script_generator=self.script_generator,
                rewrite_controller=self.rewrite_controller,
                llm_optimizer=self.llm_optimizer
            )
            logger.info("ScriptGenAPI initialized in direct mode")
        else:
            logger.info(f"ScriptGenAPI initialized with base URL: {base_url}")
    
    def get_available_llms(self) -> Dict[str, Any]:
        """
        Get list of available LLM providers and models.
        
        Returns:
            Dictionary with provider and model information
        """
        if self.direct_mode:
            # In direct mode, return the models available to the LLM wrapper
            # This is a simplified implementation - in a real app, we'd query
            # the actual models available from the providers
            return {
                "providers": [
                    {
                        "id": "openai",
                        "name": "OpenAI",
                        "models": [
                            {
                                "id": "gpt-4",
                                "name": "GPT-4",
                                "capabilities": ["high_quality", "creative", "factual"],
                                "max_tokens": 8192
                            },
                            {
                                "id": "gpt-3.5-turbo",
                                "name": "GPT-3.5 Turbo",
                                "capabilities": ["fast", "cost_effective"],
                                "max_tokens": 4096
                            }
                        ]
                    }
                ],
                "default_provider": "openai",
                "default_model": "gpt-4"
            }
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def get_all_bible_entries(
        self, 
        entry_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all bible entries, optionally filtered by type.
        
        Args:
            entry_type: Optional filter by entry type
            
        Returns:
            List of bible entries
        """
        if self.direct_mode:
            # Get entries directly from bible storage
            entries = self.bible_storage.get_all_entries(entry_type=entry_type)
            return entries
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def get_bible_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific bible entry by ID.
        
        Args:
            entry_id: ID of the bible entry
            
        Returns:
            Bible entry data or None if not found
        """
        if self.direct_mode:
            # Get entry directly from bible storage
            entry = self.bible_storage.get_entry(entry_id)
            return entry
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def create_bible_entry(
        self, 
        entry_type: str, 
        content: Dict[str, Any]
    ) -> int:
        """
        Create a new bible entry.
        
        Args:
            entry_type: Type of bible entry (character, plot, setting, theme)
            content: Content of the bible entry
            
        Returns:
            ID of the new entry
        """
        if self.direct_mode:
            # Create entry directly in bible storage
            entry_id = self.bible_storage.create_entry(entry_type, content)
            return entry_id
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def update_bible_entry(
        self, 
        entry_id: int, 
        entry_type: str, 
        content: Dict[str, Any]
    ) -> bool:
        """
        Update an existing bible entry.
        
        Args:
            entry_id: ID of the entry to update
            entry_type: Type of bible entry (character, plot, setting, theme)
            content: New content for the bible entry
            
        Returns:
            True if successful, False otherwise
        """
        if self.direct_mode:
            # Update entry directly in bible storage
            success = self.bible_storage.update_entry(entry_id, entry_type, content)
            return success
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def delete_bible_entry(self, entry_id: int) -> bool:
        """
        Delete a bible entry.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.direct_mode:
            # Delete entry directly from bible storage
            success = self.bible_storage.delete_entry(entry_id)
            return success
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def generate_scene(
        self,
        scene_id: str,
        log_line: str,
        character_ids: List[int],
        plot_id: Optional[int] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a new scene.
        
        Args:
            scene_id: ID for the new scene
            log_line: Log line describing the scene
            character_ids: List of character bible IDs
            plot_id: Optional plot bible ID
            llm_config: Optional LLM configuration
            
        Returns:
            Generated scene data
        """
        if self.direct_mode:
            # Configure LLM if needed
            if llm_config:
                self.llm_wrapper.update_config(llm_config)
            
            # Use the workflow controller for optimized generation
            existing_bibles = {
                'character': character_ids,
                'plot': [plot_id] if plot_id else []
            }
            
            # Create configuration
            config = GenerationConfig(
                scope=GenerationScope.SINGLE_SCENE,
                character_count=len(character_ids),
                scene_count=1,
                use_existing_bibles=True,
                auto_rewrite=True,
                rewrite_iterations=1,
                parallel_generation=True,
                optimize_llm_calls=True
            )
            
            # Generate the script using the workflow controller
            result = self.workflow_controller.generate_script(
                log_line=log_line,
                config=config,
                existing_bibles=existing_bibles
            )
            
            # Extract the single scene from the result
            if 'scenes' in result and len(result['scenes']) > 0:
                scene = result['scenes'][0]
                # Add the scene_id if it wasn't set
                if 'scene_id' not in scene:
                    scene['scene_id'] = scene_id
                return scene
            else:
                # Fallback to original method
                logger.warning("Workflow controller failed, falling back to agent framework")
                return self._legacy_generate_scene(
                    scene_id, log_line, character_ids, plot_id, llm_config
                )
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def _legacy_generate_scene(
        self,
        scene_id: str,
        log_line: str,
        character_ids: List[int],
        plot_id: Optional[int] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Legacy method to generate a scene using direct agent calls.
        
        Args:
            scene_id: ID for the new scene
            log_line: Log line describing the scene
            character_ids: List of character bible IDs
            plot_id: Optional plot bible ID
            llm_config: Optional LLM configuration
            
        Returns:
            Generated scene data
        """
        # Get characters and plot from bible storage
        characters = [self.bible_storage.get_entry(char_id) for char_id in character_ids]
        
        plot_bible = None
        if plot_id:
            plot_bible = self.bible_storage.get_entry(plot_id)
        
        # Set up scene context
        scene_context = {
            "log_line": log_line,
            "description": log_line
        }
        
        # Generate scene using agents
        scene = run_agents(
            scene_id=scene_id,
            characters=characters,
            plot_bible=plot_bible,
            scene_context=scene_context,
            bible_storage=self.bible_storage,
            llm_wrapper=self.llm_wrapper
        )
        
        return scene
    
    def generate_script(
        self,
        log_line: str,
        character_ids: List[int],
        plot_id: Optional[int] = None,
        scene_count: int = 3,
        use_full_script: bool = False,
        llm_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete script with multiple scenes.
        
        Args:
            log_line: Log line describing the script
            character_ids: List of character bible IDs
            plot_id: Optional plot bible ID
            scene_count: Number of scenes to generate
            use_full_script: Whether to use full script generation with act structure
            llm_config: Optional LLM configuration
            
        Returns:
            Generated script data
        """
        if self.direct_mode:
            # Configure LLM if needed
            if llm_config:
                self.llm_wrapper.update_config(llm_config)
            
            # Use existing bibles
            existing_bibles = {
                'character': character_ids,
                'plot': [plot_id] if plot_id else []
            }
            
            # Determine generation scope
            scope = GenerationScope.FULL_SCRIPT if use_full_script else GenerationScope.MULTI_SCENE
            
            # Create configuration
            config = GenerationConfig(
                scope=scope,
                character_count=len(character_ids),
                scene_count=scene_count,
                use_existing_bibles=True,
                auto_rewrite=True,
                rewrite_iterations=1,
                parallel_generation=True,
                optimize_llm_calls=True
            )
            
            # Generate the script using the workflow controller
            result = self.workflow_controller.generate_script(
                log_line=log_line,
                config=config,
                existing_bibles=existing_bibles
            )
            
            return result
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def critique_scene(
        self, 
        scene_id: str, 
        scene_content: Dict[str, Any]
    ) -> SceneCritique:
        """
        Get critique of a scene from the Dramaturge Agent.
        
        Args:
            scene_id: ID of the scene
            scene_content: Content of the scene
            
        Returns:
            SceneCritique object with structured critique
        """
        if self.direct_mode:
            # Call the rewrite controller directly
            critique = self.rewrite_controller.critique_scene(scene_id, scene_content)
            return critique
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def rewrite_scene(
        self, 
        scene_id: str, 
        scene_content: Dict[str, Any],
        user_input: str,
        regenerate_all: bool = False
    ) -> Dict[str, Any]:
        """
        Rewrite a scene based on user instructions.
        
        Args:
            scene_id: ID of the scene to rewrite
            scene_content: Current content of the scene
            user_input: User instructions for rewriting
            regenerate_all: Whether to regenerate the entire scene
            
        Returns:
            Dictionary with new scene content and diff information
        """
        if self.direct_mode:
            # Call the rewrite controller directly
            result = self.rewrite_controller.rewrite_scene(
                scene_id=scene_id,
                scene_content=scene_content,
                user_input=user_input,
                regenerate_all=regenerate_all
            )
            return result
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def propagate_change(self, change_request: ChangeRequest) -> ChangePlan:
        """
        Preview bible changes to affected scenes.
        
        Args:
            change_request: Details of the requested change
            
        Returns:
            ChangePlan with affected scenes and preview information
        """
        if self.direct_mode:
            # Call the rewrite controller directly
            plan = self.rewrite_controller.propagate_change(change_request)
            return plan
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def apply_propagation(
        self, 
        change_plan: ChangePlan,
        approved_scene_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Apply bible changes to approved scenes.
        
        Args:
            change_plan: The change plan with affected scenes
            approved_scene_ids: List of scene IDs approved for change
            
        Returns:
            Dictionary with results of the applied changes
        """
        if self.direct_mode:
            # Call the rewrite controller directly
            results = self.rewrite_controller.apply_propagation(
                change_plan=change_plan,
                approved_scene_ids=approved_scene_ids
            )
            return results
        else:
            # Make REST API call
            raise NotImplementedError("REST API mode not yet implemented")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics on LLM optimization.
        
        Returns:
            Dictionary with optimization statistics
        """
        if self.direct_mode and hasattr(self, 'llm_optimizer'):
            stats = self.llm_optimizer.get_stats()
            return {
                'cache_hits': stats.cache_hits,
                'cache_misses': stats.cache_misses,
                'deduplicated_calls': stats.deduplicated_calls,
                'batched_calls': stats.batched_calls,
                'total_calls': stats.calls,
                'total_tokens': stats.tokens,
                'estimated_cost': self.llm_optimizer.estimate_cost('gpt-4', stats.tokens)
            }
        else:
            return {
                'cache_hits': 0,
                'cache_misses': 0,
                'deduplicated_calls': 0,
                'batched_calls': 0,
                'total_calls': 0,
                'total_tokens': 0,
                'estimated_cost': 0.0
            } 