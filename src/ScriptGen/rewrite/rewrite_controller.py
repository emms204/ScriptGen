"""
Rewrite Controller for iterative script refinement.

This module implements the core functionality for the Iterative Rewrite Engine:
- Scene critique using the Dramaturge Agent
- Scene rewriting based on user instructions
- Bible change propagation across dependent scenes
"""

import json
import logging
import difflib
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field

from ..bibles.bible_storage import BibleStorage
from ..llm.llm_wrapper import LLMWrapper
from ..agents.agent_framework import DramaturgeAgent, run_agents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('rewrite_controller')


class ChangeType(Enum):
    """Types of bible changes that can be propagated"""
    NAME = "name"                  # Character name change
    ATTRIBUTE = "attribute"        # Character/Setting/Plot attribute change
    RELATIONSHIP = "relationship"  # Change in relationship between entities
    PLOT_POINT = "plot_point"      # Change in plot point or arc
    SETTING = "setting"            # Change in setting details


@dataclass
class SceneCritique:
    """Container for dramaturge critique of a scene"""
    scene_id: str
    overall_assessment: str
    pacing: Dict[str, Any]
    tension: Dict[str, Any]
    subtext: Dict[str, Any]
    character_authenticity: Dict[str, Any]
    plot_advancement: Dict[str, Any]
    suggested_rewrites: List[str]
    raw_critique: str


@dataclass
class ChangeRequest:
    """Request to change a bible entry with potential propagation"""
    bible_id: int
    bible_type: str
    change_type: ChangeType
    field_path: str  # JSON path to the changed field (e.g., "content.name")
    old_value: Any
    new_value: Any
    description: str = ""
    propagate: bool = True


@dataclass
class AffectedScene:
    """Information about a scene affected by a bible change"""
    scene_id: str
    title: str
    current_content: Dict[str, Any]
    preview_content: Optional[Dict[str, Any]] = None
    diff: Optional[str] = None
    approved: bool = False


@dataclass
class ChangePlan:
    """Plan for propagating changes across the script"""
    request: ChangeRequest
    affected_scenes: List[AffectedScene] = field(default_factory=list)
    status: str = "pending"  # pending, previewed, applied, rejected
    error: Optional[str] = None


class RewriteController:
    """
    Controller for iterative script rewriting and change propagation.
    
    This class provides methods for:
    1. Critiquing scenes using the Dramaturge Agent
    2. Rewriting scenes based on user instructions
    3. Propagating bible changes to affected scenes
    """
    
    def __init__(
        self,
        bible_storage: Optional[BibleStorage] = None,
        llm_wrapper: Optional[LLMWrapper] = None
    ):
        """
        Initialize the rewrite controller.
        
        Args:
            bible_storage: Bible storage instance
            llm_wrapper: LLM wrapper instance
        """
        self.bible_storage = bible_storage or BibleStorage()
        self.llm_wrapper = llm_wrapper or LLMWrapper()
        self.dramaturge = DramaturgeAgent(llm_wrapper=self.llm_wrapper)
        
    def critique_scene(self, scene_id: str, scene_content: Dict[str, Any]) -> SceneCritique:
        """
        Analyze a scene for pacing, tension, subtext, and other dramatic elements.
        
        Args:
            scene_id: ID of the scene to critique
            scene_content: Content of the scene (including integrated_scene)
            
        Returns:
            SceneCritique object with structured critique
        """
        logger.info(f"Critiquing scene: {scene_id}")
        
        # Get related plot bible for context
        plot_bible = self._get_related_plot(scene_id, scene_content)
        
        # Get the integrated scene text
        scene_text = scene_content.get('integrated_scene', '')
        if not scene_text:
            raise ValueError("Scene content must include 'integrated_scene'")
        
        # Run dramaturge agent to critique the scene
        critique_result = self.dramaturge.critique_scene(scene_text, plot_bible)
        critique_text = critique_result.get('critique', '')
        
        # Parse the structured critique
        # Note: This parsing assumes a certain format from the dramaturge agent output
        # In a production system, we would use a more robust parsing approach
        parsed = self._parse_critique(critique_text)
        
        return SceneCritique(
            scene_id=scene_id,
            overall_assessment=parsed.get('overall', ''),
            pacing=parsed.get('pacing', {}),
            tension=parsed.get('tension', {}),
            subtext=parsed.get('subtext', {}),
            character_authenticity=parsed.get('character_authenticity', {}),
            plot_advancement=parsed.get('plot_advancement', {}),
            suggested_rewrites=parsed.get('suggested_rewrites', []),
            raw_critique=critique_text
        )
    
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
        logger.info(f"Rewriting scene {scene_id} with instructions: {user_input}")
        
        # Get the characters and plot bible
        character_ids = scene_content.get('character_ids', [])
        characters = [self.bible_storage.get_entry(char_id) for char_id in character_ids]
        
        plot_bible = self._get_related_plot(scene_id, scene_content)
        scene_context = scene_content.get('outline', {})
        
        # Add user instructions to the scene context
        enhanced_context = dict(scene_context)
        enhanced_context['rewrite_instructions'] = user_input
        
        # If not regenerating everything, keep some parts of the original
        if not regenerate_all and 'character_dialogue' in scene_content:
            # Prepare a special prompt for the LLM wrapper to modify existing dialogue
            # This is a simplified approach; in a real implementation, we'd have
            # more sophisticated ways to selectively update parts of the scene
            existing_scene = scene_content.get('integrated_scene', '')
            prompt = self._create_rewrite_prompt(existing_scene, user_input)
            
            response = self.llm_wrapper.call_llm(
                prompt=prompt,
                params={'temperature': 0.7, 'max_tokens': 2000}
            )
            
            # Parse the response into a new scene
            new_scene_text = response.get('text', '')
            
            # Create a modified version of the original content
            new_content = dict(scene_content)
            new_content['integrated_scene'] = new_scene_text
            new_content['rewrite_instructions'] = user_input
            
        else:
            # Regenerate the entire scene with the enhanced context
            new_content = run_agents(
                scene_id=scene_id,
                characters=characters,
                plot_bible=plot_bible,
                scene_context=enhanced_context,
                bible_storage=self.bible_storage,
                llm_wrapper=self.llm_wrapper
            )
        
        # Generate diff between old and new scenes
        old_scene = scene_content.get('integrated_scene', '')
        new_scene = new_content.get('integrated_scene', '')
        diff = self._generate_diff(old_scene, new_scene)
        
        return {
            'scene_id': scene_id,
            'old_content': scene_content,
            'new_content': new_content,
            'diff': diff,
            'user_input': user_input
        }
    
    def propagate_change(self, change_request: ChangeRequest) -> ChangePlan:
        """
        Identify scenes affected by a bible change and prepare a change plan.
        
        Args:
            change_request: Details of the requested change
            
        Returns:
            ChangePlan with affected scenes and preview information
        """
        logger.info(f"Propagating change for bible ID {change_request.bible_id}")
        
        # Create change plan
        plan = ChangePlan(request=change_request)
        
        try:
            # Get the bible entry
            bible_entry = self.bible_storage.get_entry(change_request.bible_id)
            if not bible_entry:
                raise ValueError(f"Bible entry {change_request.bible_id} not found")
            
            # Apply the change to create a preview of the new bible entry
            updated_bible = self._apply_bible_change(bible_entry, change_request)
            
            # Find affected scenes
            affected_scenes = self._find_affected_scenes(change_request, bible_entry)
            logger.info(f"Found {len(affected_scenes)} affected scenes")
            
            # Generate previews for each affected scene
            for scene_id, scene_data in affected_scenes.items():
                # Create preview of the updated scene
                scene_preview = self._preview_scene_update(
                    scene_id=scene_id,
                    scene_data=scene_data,
                    change_request=change_request,
                    updated_bible=updated_bible
                )
                
                # Add to the change plan
                affected_scene = AffectedScene(
                    scene_id=scene_id,
                    title=scene_data.get('title', f"Scene {scene_id}"),
                    current_content=scene_data,
                    preview_content=scene_preview.get('new_content', None),
                    diff=scene_preview.get('diff', None)
                )
                plan.affected_scenes.append(affected_scene)
            
            plan.status = "previewed"
            
        except Exception as e:
            logger.error(f"Error propagating change: {e}")
            plan.status = "error"
            plan.error = str(e)
        
        return plan
    
    def apply_propagation(
        self, 
        change_plan: ChangePlan,
        approved_scene_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Apply the propagated changes to the approved scenes.
        
        Args:
            change_plan: The change plan with affected scenes
            approved_scene_ids: List of scene IDs that were approved for change
            
        Returns:
            Dictionary with results of the applied changes
        """
        logger.info(f"Applying propagation plan for {len(change_plan.affected_scenes)} scenes")
        
        approved_ids = set(approved_scene_ids or [])
        results = {
            'bible_updated': False,
            'scenes_updated': [],
            'scenes_skipped': []
        }
        
        try:
            # Update the bible entry
            bible_entry = self.bible_storage.get_entry(change_plan.request.bible_id)
            updated_bible = self._apply_bible_change(bible_entry, change_plan.request)
            self.bible_storage.update_entry(
                entry_id=change_plan.request.bible_id,
                entry_type=change_plan.request.bible_type,
                content=updated_bible.get('content', {})
            )
            results['bible_updated'] = True
            
            # Update approved scenes
            for scene in change_plan.affected_scenes:
                if scene.scene_id in approved_ids:
                    # In a real implementation, this would update the scene in the database
                    # For this example, we'll just track which scenes would be updated
                    results['scenes_updated'].append(scene.scene_id)
                else:
                    results['scenes_skipped'].append(scene.scene_id)
            
            change_plan.status = "applied"
            
        except Exception as e:
            logger.error(f"Error applying propagation: {e}")
            change_plan.status = "error"
            change_plan.error = str(e)
            results['error'] = str(e)
        
        return results
    
    def _parse_critique(self, critique_text: str) -> Dict[str, Any]:
        """
        Parse the raw critique text into structured data.
        
        Args:
            critique_text: Raw critique text from dramaturge agent
            
        Returns:
            Structured critique data
        """
        # This is a simplistic parsing approach
        # In a production system, we would use a more robust approach with LLM-based parsing
        
        # Initialize result structure
        result = {
            'overall': '',
            'pacing': {'issues': [], 'suggestions': []},
            'tension': {'issues': [], 'suggestions': []},
            'subtext': {'issues': [], 'suggestions': []},
            'character_authenticity': {'issues': [], 'suggestions': []},
            'plot_advancement': {'issues': [], 'suggestions': []},
            'suggested_rewrites': []
        }
        
        # Simple parsing based on section headers
        current_section = 'overall'
        for line in critique_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if "OVERALL ASSESSMENT" in line.upper():
                current_section = 'overall'
                continue
                
            if "PACING" in line.upper():
                current_section = 'pacing'
                continue
                
            if "TENSION" in line.upper():
                current_section = 'tension'
                continue
                
            if "SUBTEXT" in line.upper():
                current_section = 'subtext'
                continue
                
            if "CHARACTER AUTHENTICITY" in line.upper():
                current_section = 'character_authenticity'
                continue
                
            if "PLOT ADVANCEMENT" in line.upper():
                current_section = 'plot_advancement'
                continue
                
            if "SUGGESTED REWRITES" in line.upper() or "REWRITTEN DIALOGUE" in line.upper():
                current_section = 'suggested_rewrites'
                continue
            
            # Add content to the current section
            if current_section == 'overall':
                result['overall'] += line + " "
            elif current_section == 'suggested_rewrites':
                if line.startswith('-') or line[0].isdigit():
                    result['suggested_rewrites'].append(line)
            elif current_section in result:
                if "issue" in line.lower() or "problem" in line.lower():
                    result[current_section]['issues'].append(line)
                elif "suggest" in line.lower() or "could" in line.lower() or "should" in line.lower():
                    result[current_section]['suggestions'].append(line)
        
        return result
    
    def _create_rewrite_prompt(self, existing_scene: str, user_input: str) -> str:
        """
        Create a prompt for rewriting a scene based on user instructions.
        
        Args:
            existing_scene: Current scene text
            user_input: User instructions for rewriting
            
        Returns:
            Prompt for the LLM
        """
        return f"""
You are a skilled screenplay editor tasked with rewriting a scene based on specific instructions.

ORIGINAL SCENE:
{existing_scene}

USER INSTRUCTIONS:
{user_input}

Rewrite the scene to implement the user's instructions while maintaining the original's structure and key plot points.
Focus specifically on addressing the user's requests, making minimal changes to parts that don't need modification.

Your rewrite should follow proper screenplay format with scene headings (INT/EXT), action descriptions, and character dialogue.
"""
    
    def _generate_diff(self, old_text: str, new_text: str) -> str:
        """
        Generate a readable diff between old and new text.
        
        Args:
            old_text: Original text
            new_text: New text
            
        Returns:
            Formatted diff
        """
        diff = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            lineterm='',
            n=3  # Context lines
        )
        return '\n'.join(diff)
    
    def _apply_bible_change(self, bible_entry: Dict[str, Any], change: ChangeRequest) -> Dict[str, Any]:
        """
        Apply a change to a bible entry.
        
        Args:
            bible_entry: The bible entry to update
            change: The change request
            
        Returns:
            Updated bible entry
        """
        updated_bible = dict(bible_entry)
        
        # Parse the field path
        path_parts = change.field_path.split('.')
        
        # Navigate to the correct nested dictionary
        current = updated_bible
        for i, part in enumerate(path_parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Update the field
        last_part = path_parts[-1]
        if last_part in current:
            current[last_part] = change.new_value
        
        return updated_bible
    
    def _get_related_plot(self, scene_id: str, scene_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the plot bible related to a scene.
        
        Args:
            scene_id: ID of the scene
            scene_content: Content of the scene
            
        Returns:
            Plot bible for the scene
        """
        # First try to get from scene content
        plot_id = scene_content.get('plot_id')
        if plot_id:
            plot_bible = self.bible_storage.get_entry(plot_id)
            if plot_bible:
                return plot_bible
        
        # If not found, create a minimal plot bible from scene content
        return {
            'id': 0,
            'type': 'plot',
            'content': {
                'scene_id': scene_id,
                'description': scene_content.get('outline', {}).get('description', ''),
                'prior_state': scene_content.get('outline', {}).get('prior_state', ''),
                'next_state': scene_content.get('outline', {}).get('next_state', '')
            }
        }
    
    def _find_affected_scenes(
        self, 
        change: ChangeRequest, 
        bible_entry: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find scenes affected by a bible change.
        
        Args:
            change: The change request
            bible_entry: The bible entry being changed
            
        Returns:
            Dictionary of scene_id to scene data for affected scenes
        """
        # In a real implementation, this would query the database for scenes
        # related to the bible entry. For this example, we'll simulate finding scenes.
        
        # This is a simplified implementation
        # In a real system, we would have a more sophisticated scene database
        affected_scenes = {}
        
        # Simulate finding scenes based on bible type and change type
        if change.bible_type == "character":
            # Find scenes that include this character
            character_name = bible_entry.get('content', {}).get('name', '')
            
            # Simulate scene search results
            # In a real implementation, this would query a database
            if character_name:
                # For this example, we'll create a single dummy scene
                affected_scenes["scene_1"] = {
                    "title": f"Scene with {character_name}",
                    "outline": {
                        "description": f"A scene featuring {character_name}",
                        "objectives": ["Reveal character motivation"]
                    },
                    "integrated_scene": f"""INT. CAFE - DAY

{character_name} sits at a table, nervously checking their watch.

{character_name}
(anxious)
Where could they be? They're never late.

A door opens, and {character_name} looks up expectantly.
""",
                    "character_ids": [change.bible_id]
                }
        
        elif change.bible_type == "plot":
            # Find scenes related to this plot
            plot_id = bible_entry.get('id', 0)
            
            # Simulate scene search results
            affected_scenes["scene_2"] = {
                "title": "Plot-related scene",
                "outline": {
                    "description": "A crucial plot scene",
                    "objectives": ["Advance the main plot"]
                },
                "integrated_scene": "INT. HEADQUARTERS - NIGHT\n\nThe team reviews the evidence board.",
                "plot_id": plot_id
            }
        
        return affected_scenes
    
    def _preview_scene_update(
        self,
        scene_id: str,
        scene_data: Dict[str, Any],
        change_request: ChangeRequest,
        updated_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a preview of how a scene would be updated after a bible change.
        
        Args:
            scene_id: ID of the scene
            scene_data: Current scene data
            change_request: The change request
            updated_bible: The updated bible entry
            
        Returns:
            Dictionary with preview information
        """
        # Simple text replacement for name changes
        if change_request.change_type == ChangeType.NAME:
            old_name = change_request.old_value
            new_name = change_request.new_value
            
            old_scene = scene_data.get('integrated_scene', '')
            new_scene = old_scene.replace(old_name, new_name)
            
            # Create updated scene data
            new_content = dict(scene_data)
            new_content['integrated_scene'] = new_scene
            
            # Generate diff
            diff = self._generate_diff(old_scene, new_scene)
            
            return {
                'scene_id': scene_id,
                'old_content': scene_data,
                'new_content': new_content,
                'diff': diff
            }
        
        # For more complex changes, we would regenerate the scene using agents
        # This is simplified for this example
        return {
            'scene_id': scene_id,
            'old_content': scene_data,
            'new_content': scene_data,  # No change for now
            'diff': "# No changes previewed for complex change types\n# Would regenerate scene in production"
        } 