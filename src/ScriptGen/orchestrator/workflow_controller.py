"""
Workflow Controller for managing the script generation pipeline.

This module provides the central controller that ties all modules into a seamless
pipeline, orchestrating the full script generation process from log line to final script.
"""

import os
import json
import logging
import time
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..bibles.bible_storage import BibleStorage
from ..llm.llm_wrapper import LLMWrapper
from ..generators.script_generator import ScriptGenerator
from ..agents.agent_framework import run_agents
from ..rewrite.rewrite_controller import RewriteController
from .optimizer import LLMOptimizer, CacheConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('workflow_controller')


class WorkflowStep(Enum):
    """Enumeration of steps in the workflow process"""
    INITIALIZE = auto()
    BOOTSTRAP_BIBLES = auto()
    GENERATE_SCENE_OUTLINE = auto()
    RUN_AGENTS = auto()
    CRITIQUE_SCENE = auto()
    REWRITE_SCENE = auto()
    FINALIZE = auto()


class GenerationScope(Enum):
    """Scope of script generation"""
    SINGLE_SCENE = auto()
    MULTI_SCENE = auto()
    FULL_SCRIPT = auto()


class GenerationConfig:
    """Configuration for script generation"""
    
    def __init__(
        self,
        scope: GenerationScope = GenerationScope.SINGLE_SCENE,
        character_count: int = 2,
        scene_count: int = 1,
        use_existing_bibles: bool = False,
        auto_rewrite: bool = False,
        rewrite_iterations: int = 1,
        parallel_generation: bool = True,
        optimize_llm_calls: bool = True,
        cache_config: Optional[CacheConfig] = None
    ):
        """
        Initialize generation configuration.
        
        Args:
            scope: Generation scope (scene, multi-scene, full script)
            character_count: Number of characters to create/use
            scene_count: Number of scenes to generate
            use_existing_bibles: Whether to use existing bibles or create new ones
            auto_rewrite: Whether to automatically rewrite based on critique
            rewrite_iterations: Number of rewrite iterations to perform
            parallel_generation: Whether to use parallel generation for characters/scenes
            optimize_llm_calls: Whether to use LLM optimization
            cache_config: Cache configuration for LLM optimization
        """
        self.scope = scope
        self.character_count = character_count
        self.scene_count = scene_count
        self.use_existing_bibles = use_existing_bibles
        self.auto_rewrite = auto_rewrite
        self.rewrite_iterations = rewrite_iterations
        self.parallel_generation = parallel_generation
        self.optimize_llm_calls = optimize_llm_calls
        self.cache_config = cache_config or CacheConfig()


class WorkflowController:
    """
    Central controller that orchestrates the script generation process.
    
    This class integrates all modules into a seamless pipeline:
    1. Initializes required components
    2. Bootstraps bible entries if needed
    3. Generates scene outlines
    4. Runs agents to create dialogue and integrated scenes
    5. Provides critique and rewrite capabilities
    6. Finalizes and returns the generated script
    """
    
    def __init__(
        self,
        bible_storage: Optional[BibleStorage] = None,
        llm_wrapper: Optional[LLMWrapper] = None,
        script_generator: Optional[ScriptGenerator] = None,
        rewrite_controller: Optional[RewriteController] = None,
        llm_optimizer: Optional[LLMOptimizer] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the workflow controller.
        
        Args:
            bible_storage: Bible storage instance
            llm_wrapper: LLM wrapper instance
            script_generator: Script generator instance
            rewrite_controller: Rewrite controller instance
            llm_optimizer: LLM optimizer instance
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize components
        self.bible_storage = bible_storage or BibleStorage()
        self.llm_wrapper = llm_wrapper or LLMWrapper()
        self.script_generator = script_generator or ScriptGenerator(
            bible_storage=self.bible_storage,
            llm_wrapper=self.llm_wrapper
        )
        self.rewrite_controller = rewrite_controller or RewriteController(
            bible_storage=self.bible_storage,
            llm_wrapper=self.llm_wrapper
        )
        self.llm_optimizer = llm_optimizer or LLMOptimizer(llm_wrapper=self.llm_wrapper)
        
        # Workflow tracking
        self.current_step = WorkflowStep.INITIALIZE
        self.stats = {
            'start_time': None,
            'end_time': None,
            'llm_calls': 0,
            'tokens_used': 0,
            'scenes_generated': 0,
            'rewrite_iterations': 0
        }
    
    def generate_script(
        self,
        log_line: str,
        config: Optional[GenerationConfig] = None,
        existing_bibles: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a script based on a log line.
        
        Args:
            log_line: Log line describing the script
            config: Generation configuration
            existing_bibles: IDs of existing bibles to use
            
        Returns:
            Generated script data
        """
        # Initialize or use provided configuration
        config = config or GenerationConfig()
        
        # Start tracking stats
        self.stats['start_time'] = time.time()
        self.stats['llm_calls'] = 0
        self.stats['tokens_used'] = 0
        self.stats['scenes_generated'] = 0
        self.stats['rewrite_iterations'] = 0
        
        try:
            # Step 1: Bootstrap bibles if needed
            self.current_step = WorkflowStep.BOOTSTRAP_BIBLES
            bible_ids = self._bootstrap_bibles(log_line, config, existing_bibles)
            
            # Step 2: Process script based on scope
            processor = ScriptProcessor(
                bible_storage=self.bible_storage,
                llm_wrapper=self.llm_wrapper,
                script_generator=self.script_generator,
                rewrite_controller=self.rewrite_controller,
                llm_optimizer=self.llm_optimizer if config.optimize_llm_calls else None
            )
            
            # Generate script based on scope
            result = processor.process_script(log_line, bible_ids, config)
            
            # Update stats
            self.stats['scenes_generated'] = len(result.get('scenes', []))
            self.stats['end_time'] = time.time()
            
            # Add stats to result
            result['stats'] = self._get_stats()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in script generation: {e}")
            self.stats['end_time'] = time.time()
            return {
                'error': str(e),
                'stats': self._get_stats(),
                'step': self.current_step.name
            }
    
    def _bootstrap_bibles(
        self,
        log_line: str,
        config: GenerationConfig,
        existing_bibles: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, List[int]]:
        """
        Bootstrap bible entries based on log line.
        
        Args:
            log_line: Log line describing the script
            config: Generation configuration
            existing_bibles: IDs of existing bibles to use
            
        Returns:
            Dictionary of bible IDs by type
        """
        # If using existing bibles, return them
        if config.use_existing_bibles and existing_bibles:
            return existing_bibles
        
        # Otherwise, create new bibles
        logger.info(f"Bootstrapping bibles for log line: {log_line}")
        
        # Generate character bibles
        character_ids = []
        num_characters = config.character_count
        
        # Generate character bibles using script generator
        for i in range(num_characters):
            character_data = self.script_generator.bootstrap_bible(
                log_line, 
                bible_type='character',
                sequence_number=i
            )
            character_id = self.bible_storage.create_entry('character', character_data)
            character_ids.append(character_id)
            
            # Update LLM call stats
            self.stats['llm_calls'] += 1
        
        # Generate plot bible
        plot_data = self.script_generator.bootstrap_bible(log_line, bible_type='plot')
        plot_id = self.bible_storage.create_entry('plot', plot_data)
        
        # Update LLM call stats
        self.stats['llm_calls'] += 1
        
        # Generate setting bible if needed
        setting_data = self.script_generator.bootstrap_bible(log_line, bible_type='setting')
        setting_id = self.bible_storage.create_entry('setting', setting_data)
        
        # Update LLM call stats
        self.stats['llm_calls'] += 1
        
        return {
            'character': character_ids,
            'plot': [plot_id],
            'setting': [setting_id]
        }
    
    def _get_stats(self) -> Dict[str, Any]:
        """
        Get current workflow statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            stats['duration'] = stats['end_time'] - stats['start_time']
        
        return stats


class ScriptProcessor:
    """
    Processor for script generation based on scope.
    
    This class handles the generation of scripts at different scopes:
    1. Single scene: Generates a single scene
    2. Multi-scene: Generates multiple connected scenes
    3. Full script: Generates a complete script with act structure
    """
    
    def __init__(
        self,
        bible_storage: BibleStorage,
        llm_wrapper: LLMWrapper,
        script_generator: ScriptGenerator,
        rewrite_controller: RewriteController,
        llm_optimizer: Optional[LLMOptimizer] = None
    ):
        """
        Initialize the script processor.
        
        Args:
            bible_storage: Bible storage instance
            llm_wrapper: LLM wrapper instance
            script_generator: Script generator instance
            rewrite_controller: Rewrite controller instance
            llm_optimizer: LLM optimizer instance
        """
        self.bible_storage = bible_storage
        self.llm_wrapper = llm_wrapper
        self.script_generator = script_generator
        self.rewrite_controller = rewrite_controller
        self.llm_optimizer = llm_optimizer
        
        # Stats tracking
        self.llm_calls = 0
        self.tokens_used = 0
    
    def process_script(
        self,
        log_line: str,
        bible_ids: Dict[str, List[int]],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Process script generation based on scope.
        
        Args:
            log_line: Log line describing the script
            bible_ids: Dictionary of bible IDs by type
            config: Generation configuration
            
        Returns:
            Generated script data
        """
        # Different processing based on scope
        if config.scope == GenerationScope.SINGLE_SCENE:
            return self._process_single_scene(log_line, bible_ids, config)
        elif config.scope == GenerationScope.MULTI_SCENE:
            return self._process_multi_scene(log_line, bible_ids, config)
        elif config.scope == GenerationScope.FULL_SCRIPT:
            return self._process_full_script(log_line, bible_ids, config)
        else:
            raise ValueError(f"Unknown generation scope: {config.scope}")
    
    def _process_single_scene(
        self,
        log_line: str,
        bible_ids: Dict[str, List[int]],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Process single scene generation.
        
        Args:
            log_line: Log line describing the scene
            bible_ids: Dictionary of bible IDs by type
            config: Generation configuration
            
        Returns:
            Generated scene data
        """
        logger.info(f"Processing single scene for log line: {log_line}")
        
        # Get bible entries
        character_ids = bible_ids.get('character', [])
        characters = [self.bible_storage.get_entry(char_id) for char_id in character_ids]
        
        plot_ids = bible_ids.get('plot', [])
        plot_bible = self.bible_storage.get_entry(plot_ids[0]) if plot_ids else None
        
        # Set up scene context
        scene_context = {
            "log_line": log_line,
            "description": log_line
        }
        
        # Generate scene using agents
        scene_id = f"scene_{int(time.time())}"
        
        # Use LLM optimizer if available
        if self.llm_optimizer and config.optimize_llm_calls:
            self.llm_optimizer.start_batch()
        
        scene = run_agents(
            scene_id=scene_id,
            characters=characters,
            plot_bible=plot_bible,
            scene_context=scene_context,
            bible_storage=self.bible_storage,
            llm_wrapper=self.llm_wrapper,
            parallel=config.parallel_generation
        )
        
        if self.llm_optimizer and config.optimize_llm_calls:
            stats = self.llm_optimizer.end_batch()
            self.llm_calls += stats.calls
            self.tokens_used += stats.tokens
        
        # Apply auto-rewrites if configured
        if config.auto_rewrite and config.rewrite_iterations > 0:
            scene = self._apply_rewrites(scene, config.rewrite_iterations)
        
        return {
            "title": f"Scene: {log_line[:50]}...",
            "scenes": [scene],
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used
        }
    
    def _process_multi_scene(
        self,
        log_line: str,
        bible_ids: Dict[str, List[int]],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Process multi-scene generation.
        
        Args:
            log_line: Log line describing the script
            bible_ids: Dictionary of bible IDs by type
            config: Generation configuration
            
        Returns:
            Generated script data with multiple scenes
        """
        logger.info(f"Processing multi-scene script for log line: {log_line}")
        
        # Get plot bible
        plot_ids = bible_ids.get('plot', [])
        plot_bible = self.bible_storage.get_entry(plot_ids[0]) if plot_ids else None
        
        # Generate scene outlines
        scene_count = config.scene_count
        scene_outlines = []
        
        # If we have a plot bible with scenes, use those
        if plot_bible and 'content' in plot_bible and 'scenes' in plot_bible['content']:
            plot_scenes = plot_bible['content']['scenes']
            for i, scene in enumerate(plot_scenes[:scene_count]):
                scene_outlines.append({
                    "scene_id": f"scene_{i+1}",
                    "description": scene.get('description', f"Scene {i+1}"),
                    "log_line": scene.get('description', f"Scene {i+1} of {log_line}")
                })
        else:
            # Generate scene outlines from log line
            for i in range(scene_count):
                # Use script generator to create scene outline
                scene_desc = self.script_generator.generate_scene_outline(
                    log_line, 
                    sequence_number=i,
                    total_scenes=scene_count
                )
                
                scene_outlines.append({
                    "scene_id": f"scene_{i+1}",
                    "description": scene_desc.get('description', f"Scene {i+1}"),
                    "log_line": scene_desc.get('description', f"Scene {i+1} of {log_line}")
                })
                
                # Update LLM call stats
                self.llm_calls += 1
        
        # Generate each scene
        scenes = []
        
        # Use parallel processing if configured
        if config.parallel_generation and len(scene_outlines) > 1:
            with ThreadPoolExecutor(max_workers=min(len(scene_outlines), 5)) as executor:
                future_to_scene = {
                    executor.submit(
                        self._generate_scene,
                        outline,
                        bible_ids,
                        config
                    ): outline for outline in scene_outlines
                }
                
                for future in future_to_scene:
                    try:
                        scene = future.result()
                        scenes.append(scene)
                    except Exception as e:
                        logger.error(f"Error generating scene: {e}")
        else:
            # Generate scenes sequentially
            for outline in scene_outlines:
                scene = self._generate_scene(outline, bible_ids, config)
                scenes.append(scene)
        
        # Sort scenes by scene_id
        scenes.sort(key=lambda s: s.get('scene_id', ''))
        
        return {
            "title": f"Script: {log_line[:50]}...",
            "scenes": scenes,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used
        }
    
    def _process_full_script(
        self,
        log_line: str,
        bible_ids: Dict[str, List[int]],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Process full script generation with act structure.
        
        Args:
            log_line: Log line describing the script
            bible_ids: Dictionary of bible IDs by type
            config: Generation configuration
            
        Returns:
            Generated full script data
        """
        logger.info(f"Processing full script for log line: {log_line}")
        
        # Get plot bible
        plot_ids = bible_ids.get('plot', [])
        plot_bible = self.bible_storage.get_entry(plot_ids[0]) if plot_ids else None
        
        # Determine act structure
        act_structure = "3-act"  # Default
        if plot_bible and 'content' in plot_bible and 'act_structure' in plot_bible['content']:
            act_structure = plot_bible['content']['act_structure']
        
        # Determine scenes per act based on act structure
        scenes_per_act = self._determine_scenes_per_act(act_structure, config.scene_count)
        
        # Generate scenes for each act
        all_scenes = []
        act_count = len(scenes_per_act)
        
        for act_num, scene_count in enumerate(scenes_per_act, 1):
            # Generate act introduction
            act_log_line = f"Act {act_num} of {act_count}: "
            if act_num == 1:
                act_log_line += f"Setup and introduction of {log_line}"
            elif act_num == act_count:
                act_log_line += f"Resolution and conclusion of {log_line}"
            else:
                act_log_line += f"Confrontation and complications of {log_line}"
            
            # Configure for this act
            act_config = GenerationConfig(
                scope=GenerationScope.MULTI_SCENE,
                character_count=config.character_count,
                scene_count=scene_count,
                use_existing_bibles=True,
                auto_rewrite=config.auto_rewrite,
                rewrite_iterations=config.rewrite_iterations,
                parallel_generation=config.parallel_generation,
                optimize_llm_calls=config.optimize_llm_calls
            )
            
            # Generate scenes for this act
            act_result = self._process_multi_scene(act_log_line, bible_ids, act_config)
            
            # Add act information to scenes
            for scene in act_result.get('scenes', []):
                scene['act'] = act_num
                all_scenes.append(scene)
            
            # Update LLM call stats
            self.llm_calls += act_result.get('llm_calls', 0)
            self.tokens_used += act_result.get('tokens_used', 0)
        
        return {
            "title": f"Full Script: {log_line[:50]}...",
            "act_structure": act_structure,
            "acts": act_count,
            "scenes": all_scenes,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used
        }
    
    def _generate_scene(
        self,
        outline: Dict[str, Any],
        bible_ids: Dict[str, List[int]],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Generate a scene from an outline.
        
        Args:
            outline: Scene outline data
            bible_ids: Dictionary of bible IDs by type
            config: Generation configuration
            
        Returns:
            Generated scene data
        """
        # Get bible entries
        character_ids = bible_ids.get('character', [])
        characters = [self.bible_storage.get_entry(char_id) for char_id in character_ids]
        
        plot_ids = bible_ids.get('plot', [])
        plot_bible = self.bible_storage.get_entry(plot_ids[0]) if plot_ids else None
        
        # Set up scene context
        scene_context = {
            "log_line": outline.get('log_line', ''),
            "description": outline.get('description', '')
        }
        
        # Use LLM optimizer if available
        if self.llm_optimizer and config.optimize_llm_calls:
            self.llm_optimizer.start_batch()
        
        # Generate scene using agents
        scene = run_agents(
            scene_id=outline.get('scene_id', f"scene_{int(time.time())}"),
            characters=characters,
            plot_bible=plot_bible,
            scene_context=scene_context,
            bible_storage=self.bible_storage,
            llm_wrapper=self.llm_wrapper,
            parallel=config.parallel_generation
        )
        
        if self.llm_optimizer and config.optimize_llm_calls:
            stats = self.llm_optimizer.end_batch()
            self.llm_calls += stats.calls
            self.tokens_used += stats.tokens
        
        # Apply auto-rewrites if configured
        if config.auto_rewrite and config.rewrite_iterations > 0:
            scene = self._apply_rewrites(scene, config.rewrite_iterations)
        
        return scene
    
    def _apply_rewrites(
        self,
        scene: Dict[str, Any],
        iterations: int
    ) -> Dict[str, Any]:
        """
        Apply automatic rewrites to a scene.
        
        Args:
            scene: Scene data to rewrite
            iterations: Number of rewrite iterations
            
        Returns:
            Rewritten scene data
        """
        current_scene = scene
        
        for i in range(iterations):
            # Get critique
            critique = self.rewrite_controller.critique_scene(
                scene_id=current_scene.get('scene_id', ''),
                scene_content=current_scene
            )
            
            # Skip rewrite if no issues found
            if not critique.has_issues():
                logger.info(f"No issues found in iteration {i+1}, skipping rewrite")
                break
            
            # Generate rewrite instructions
            rewrite_instructions = critique.generate_rewrite_instructions()
            
            # Apply rewrite
            rewrite_result = self.rewrite_controller.rewrite_scene(
                scene_id=current_scene.get('scene_id', ''),
                scene_content=current_scene,
                user_input=rewrite_instructions
            )
            
            # Update current scene
            current_scene = rewrite_result.get('new_scene', current_scene)
            
            # Update LLM call stats
            self.llm_calls += 2  # One for critique, one for rewrite
        
        return current_scene
    
    def _determine_scenes_per_act(
        self,
        act_structure: str,
        total_scenes: int
    ) -> List[int]:
        """
        Determine number of scenes per act based on structure.
        
        Args:
            act_structure: Act structure (3-act, 5-act, etc.)
            total_scenes: Total number of scenes
            
        Returns:
            List of scene counts per act
        """
        if act_structure == "3-act":
            # Approximate 25% - 50% - 25% distribution
            if total_scenes <= 3:
                return [1] * total_scenes
            first_act = max(1, total_scenes // 4)
            third_act = max(1, total_scenes // 4)
            second_act = total_scenes - first_act - third_act
            return [first_act, second_act, third_act]
        
        elif act_structure == "5-act":
            # Approximate Freytag's pyramid distribution
            if total_scenes <= 5:
                return [1] * total_scenes
            act_1 = max(1, total_scenes // 6)  # Exposition
            act_2 = max(1, total_scenes // 5)  # Rising action
            act_3 = max(1, total_scenes // 4)  # Climax
            act_4 = max(1, total_scenes // 5)  # Falling action
            act_5 = max(1, total_scenes // 6)  # Resolution
            
            # Adjust to match total
            total = act_1 + act_2 + act_3 + act_4 + act_5
            if total < total_scenes:
                act_3 += (total_scenes - total)  # Add extras to climax
            elif total > total_scenes:
                # Remove from each act until we match
                while total > total_scenes:
                    if act_5 > 1:
                        act_5 -= 1
                    elif act_1 > 1:
                        act_1 -= 1
                    elif act_4 > 1:
                        act_4 -= 1
                    elif act_2 > 1:
                        act_2 -= 1
                    elif act_3 > 1:
                        act_3 -= 1
                    total = act_1 + act_2 + act_3 + act_4 + act_5
            
            return [act_1, act_2, act_3, act_4, act_5]
        
        else:
            # Default to equal distribution
            acts = 3  # Default to 3 acts if unknown structure
            if act_structure == "4-act":
                acts = 4
            elif act_structure == "5-act":
                acts = 5
            elif act_structure == "7-act":
                acts = 7
            
            # Calculate base scenes per act and remainder
            base = total_scenes // acts
            remainder = total_scenes % acts
            
            # Distribute scenes among acts
            result = [base] * acts
            for i in range(remainder):
                result[i] += 1
            
            return result 