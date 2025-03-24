"""
Agent Framework for collaborative script generation and refinement.

This module implements specialized role-based agents that work together
to generate and refine script components:

1. Character Agent: Generates dialogue for a specific character
2. Director Agent: Integrates character dialogue into a cohesive scene
3. Dramaturge Agent: Reviews scenes for pacing, tension, and subtext
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
import concurrent.futures

from ..llm.llm_wrapper import LLMWrapper
from ..bibles.bible_storage import BibleStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('agent_framework')


class BaseAgent:
    """Base class for all agents in the framework"""
    
    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None):
        """
        Initialize the base agent.
        
        Args:
            llm_wrapper: LLM wrapper for text generation
        """
        self.llm_wrapper = llm_wrapper or LLMWrapper()
        
    def _generate_text(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate text using the LLM wrapper.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Optional specific model to use
            
        Returns:
            Response dictionary from the LLM
        """
        logger.debug(f"Generating text with prompt: {prompt[:100]}...")
        
        params = {
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        response = self.llm_wrapper.call_llm(prompt=prompt, model=model, params=params)
        return response


class CharacterAgent(BaseAgent):
    """
    Agent that generates dialogue for a specific character in a given scene context.
    """
    
    def generate_dialogue(
        self, 
        character_bible: Dict[str, Any], 
        scene_context: str,
        num_lines: int = 10
    ) -> Dict[str, Any]:
        """
        Generate dialogue for a character in a specific scene context.
        
        Args:
            character_bible: Bible entry for the character
            scene_context: Description of the scene context
            num_lines: Number of dialogue lines to generate
            
        Returns:
            Dictionary containing the generated dialogue
        """
        character_name = character_bible.get('content', {}).get('name', 'Unknown Character')
        logger.info(f"Generating dialogue for character: {character_name}")
        
        # Format the character bible content
        character_content = json.dumps(character_bible.get('content', {}), indent=2)
        
        # Construct prompt
        prompt = f"""
You are a skilled screenwriter acting as {character_name}. Generate {num_lines} lines of dialogue for this character based on their bible entry and the scene context.

CHARACTER BIBLE:
{character_content}

SCENE CONTEXT:
{scene_context}

You must stay true to the character's personality, goals, fears, and speaking style. The dialogue should feel natural and authentic to this character.

For each line, include any relevant emotion or action in parentheses, followed by the dialogue text.

Generate exactly {num_lines} lines of dialogue in this format:
1. (emotion/action) Dialogue text
2. (emotion/action) Dialogue text
...and so on.
"""
        
        # Generate dialogue
        response = self._generate_text(prompt)
        
        # Process response
        dialogue_lines = []
        raw_text = response.get('text', '')
        
        for line in raw_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Skip line numbers and any other non-dialogue content
            if line[0].isdigit() and ')' in line:
                line = line.split(')', 1)[1].strip()
                
            if line and '(' in line and ')' in line:
                action_end = line.find(')') + 1
                action = line[:action_end].strip()
                text = line[action_end:].strip()
                dialogue_lines.append({
                    'character': character_name,
                    'action': action,
                    'text': text
                })
            elif line:
                dialogue_lines.append({
                    'character': character_name,
                    'action': '',
                    'text': line
                })
        
        return {
            'character': character_name,
            'dialogue_lines': dialogue_lines,
            'raw_response': raw_text
        }


class DirectorAgent(BaseAgent):
    """
    Agent that integrates character dialogue into a cohesive scene,
    referencing the plot bible and scene outline.
    """
    
    def integrate_scene(
        self, 
        character_outputs: List[Dict[str, Any]], 
        plot_bible: Dict[str, Any],
        scene_outline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrate character dialogue into a cohesive scene.
        
        Args:
            character_outputs: List of outputs from character agents
            plot_bible: Bible entry for the plot
            scene_outline: Outline of the scene
            
        Returns:
            Dictionary containing the integrated scene
        """
        logger.info("Integrating character dialogue into cohesive scene")
        
        # Format the character outputs
        character_dialogue = ""
        for output in character_outputs:
            character_name = output.get('character', 'Unknown')
            character_dialogue += f"\n## {character_name}'s DIALOGUE:\n"
            
            for line in output.get('dialogue_lines', []):
                action = line.get('action', '')
                text = line.get('text', '')
                character_dialogue += f"{action} {text}\n"
        
        # Format the plot bible and scene outline
        plot_content = json.dumps(plot_bible.get('content', {}), indent=2)
        scene_content = json.dumps(scene_outline, indent=2)
        
        # Construct prompt
        prompt = f"""
You are a skilled film director. Integrate the following character dialogue lines into a cohesive scene, referencing the plot bible and scene outline.

PLOT BIBLE:
{plot_content}

SCENE OUTLINE:
{scene_content}

CHARACTER DIALOGUE:
{character_dialogue}

Create a properly formatted screenplay scene that:
1. Includes a scene heading (INT/EXT)
2. Includes scene description/action
3. Integrates the character dialogue in a natural flow
4. Adds any necessary transitions between dialogue
5. Maintains the plot points from the scene outline
6. Creates dramatic tension where appropriate

Format your response as a properly formatted screenplay scene with scene heading, action, and dialogue.
"""
        
        # Generate integrated scene
        response = self._generate_text(prompt)
        
        # Process response
        integrated_scene = response.get('text', '')
        
        return {
            'integrated_scene': integrated_scene,
            'raw_response': response
        }


class DramaturgeAgent(BaseAgent):
    """
    Agent that reviews scenes for pacing, tension, subtext, and suggests improvements.
    """
    
    def critique_scene(self, scene: str, plot_bible: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review a scene for pacing, tension, subtext and suggest changes.
        
        Args:
            scene: The scene to critique
            plot_bible: Bible entry for the plot
            
        Returns:
            Dictionary containing the critique and suggestions
        """
        logger.info("Critiquing scene for pacing, tension, and subtext")
        
        # Format the plot bible
        plot_content = json.dumps(plot_bible.get('content', {}), indent=2)
        
        # Construct prompt
        prompt = f"""
You are a skilled dramaturg with expertise in script analysis. Review the following scene for pacing, tension, subtext, and narrative flow. Provide specific suggestions for improvement.

PLOT BIBLE:
{plot_content}

SCENE:
{scene}

Provide a detailed critique addressing:
1. Pacing: Is the scene moving at an appropriate pace? Are there areas that drag or move too quickly?
2. Tension: Is there sufficient conflict or tension? How could it be enhanced?
3. Subtext: Is there subtext in the dialogue? How could it be improved?
4. Character authenticity: Do the characters sound distinct and true to their descriptions?
5. Plot advancement: Does the scene effectively advance the plot as described in the bible?

First provide an overall assessment, then specific suggestions organized by category. Be constructive and specific.
Finally, provide 2-3 rewritten dialogue exchanges that demonstrate your suggestions.
"""
        
        # Generate critique
        response = self._generate_text(prompt)
        
        # Process response
        critique = response.get('text', '')
        
        return {
            'critique': critique,
            'raw_response': response
        }


def run_agents(
    scene_id: str, 
    characters: List[Dict[str, Any]], 
    plot_bible: Dict[str, Any],
    scene_context: Dict[str, Any],
    bible_storage: Optional[BibleStorage] = None,
    llm_wrapper: Optional[LLMWrapper] = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Orchestrate agent calls, passing bible data as context.
    
    Args:
        scene_id: ID of the scene
        characters: List of character bible entries
        plot_bible: Plot bible entry
        scene_context: Scene context/outline
        bible_storage: Bible storage instance
        llm_wrapper: LLM wrapper instance
        parallel: Whether to run character agent calls in parallel
        
    Returns:
        Dictionary containing scene_id, outline, dialogue, and critique
    """
    logger.info(f"Running agents for scene: {scene_id}")
    
    # Initialize agents
    llm_wrapper = llm_wrapper or LLMWrapper()
    char_agent = CharacterAgent(llm_wrapper)
    director_agent = DirectorAgent(llm_wrapper)
    dramaturge_agent = DramaturgeAgent(llm_wrapper)
    
    # Step 1: Generate dialogue for each character
    character_outputs = []
    scene_context_str = json.dumps(scene_context)
    
    if parallel and len(characters) > 1:
        # Run character agent calls in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for character in characters:
                future = executor.submit(
                    char_agent.generate_dialogue,
                    character,
                    scene_context_str
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    character_outputs.append(result)
                except Exception as e:
                    logger.error(f"Error running character agent: {e}")
    else:
        # Run character agent calls sequentially
        for character in characters:
            try:
                result = char_agent.generate_dialogue(character, scene_context_str)
                character_outputs.append(result)
            except Exception as e:
                logger.error(f"Error running character agent for {character.get('content', {}).get('name', 'Unknown')}: {e}")
    
    # Step 2: Integrate dialogue into a cohesive scene
    integrated_scene = director_agent.integrate_scene(
        character_outputs,
        plot_bible,
        scene_context
    )
    
    # Step 3: Critique the scene
    scene_critique = dramaturge_agent.critique_scene(
        integrated_scene['integrated_scene'],
        plot_bible
    )
    
    # Combine results
    result = {
        'scene_id': scene_id,
        'outline': scene_context,
        'character_dialogue': character_outputs,
        'integrated_scene': integrated_scene['integrated_scene'],
        'critique': scene_critique['critique'],
    }
    
    return result


def batch_character_agent_call(
    characters: List[Dict[str, Any]], 
    scene_context: str,
    llm_wrapper: Optional[LLMWrapper] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch multiple character agent calls into one prompt with separators.
    
    Args:
        characters: List of character bible entries
        scene_context: Description of the scene context
        llm_wrapper: LLM wrapper instance
        
    Returns:
        Dictionary with character outputs
    """
    logger.info(f"Batching character agent calls for {len(characters)} characters")
    
    llm_wrapper = llm_wrapper or LLMWrapper()
    
    # Construct the batched prompt
    prompt = f"""
You are a skilled screenwriter. Generate dialogue for multiple characters in a specific scene context.

SCENE CONTEXT:
{scene_context}

For each character, generate 5-8 lines of dialogue that reflect their personality, goals, and fears.
Each line should include any relevant emotion or action in parentheses, followed by the dialogue text.

"""
    
    # Add each character to the prompt
    for i, character in enumerate(characters):
        char_name = character.get('content', {}).get('name', f'Character {i+1}')
        char_content = json.dumps(character.get('content', {}), indent=2)
        
        prompt += f"""
CHARACTER {i+1}: {char_name}
{char_content}

GENERATE DIALOGUE FOR {char_name}:
(Separate each character's dialogue with "===CHARACTER END===")

"""
    
    # Generate dialogue for all characters
    response = llm_wrapper.call_llm(
        prompt=prompt, 
        params={'temperature': 0.7, 'max_tokens': 2000}
    )
    
    # Process response by splitting at the separator
    raw_text = response.get('text', '')
    character_sections = raw_text.split("===CHARACTER END===")
    
    # Match sections to characters
    character_outputs = []
    for i, section in enumerate(character_sections):
        if i >= len(characters):
            break
            
        char_name = characters[i].get('content', {}).get('name', f'Character {i+1}')
        
        # Parse the dialogue lines
        dialogue_lines = []
        for line in section.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line and '(' in line and ')' in line:
                action_end = line.find(')') + 1
                action = line[:action_end].strip()
                text = line[action_end:].strip()
                dialogue_lines.append({
                    'character': char_name,
                    'action': action,
                    'text': text
                })
            elif line:
                dialogue_lines.append({
                    'character': char_name,
                    'action': '',
                    'text': line
                })
        
        character_outputs.append({
            'character': char_name,
            'dialogue_lines': dialogue_lines,
            'raw_response': section
        })
    
    return {
        'character_outputs': character_outputs,
        'raw_response': raw_text
    } 