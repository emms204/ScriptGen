import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('script_generator')

# Import other components
try:
    from ..bibles.bible_storage import BibleStorage
    from ..llm.llm_wrapper import LLMWrapper
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import all dependencies: {str(e)}. Using mock functionality.")
    DEPENDENCIES_AVAILABLE = False

class ScriptGenerator:
    """
    Generate script scenes based on log lines, bible entries, and LLM processing.
    """
    
    def __init__(
        self, 
        bible_storage: Optional['BibleStorage'] = None,
        llm_wrapper: Optional['LLMWrapper'] = None,
        prompts_path: Optional[str] = None
    ):
        """
        Initialize the script generator.
        
        Args:
            bible_storage: Instance of BibleStorage
            llm_wrapper: Instance of LLMWrapper
            prompts_path: Path to prompts JSON file
        """
        # Default prompts path if not provided
        if prompts_path is None:
            prompts_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config',
                'prompts.json'
            )
        
        # Load prompts
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r') as f:
                self.prompts = json.load(f)
        else:
            logger.warning(f"Prompts file not found at {prompts_path}. Using default prompts.")
            self.prompts = self._get_default_prompts()
            
        # Initialize dependencies
        if not DEPENDENCIES_AVAILABLE:
            bible_storage = self._create_mock_bible_storage()
            llm_wrapper = self._create_mock_llm_wrapper()
            logger.warning("Using mock dependencies. Full functionality unavailable.")
        else:
            # Create instances if not provided
            self.bible_storage = bible_storage or BibleStorage()
            self.llm_wrapper = llm_wrapper or LLMWrapper()
    
    def _create_mock_bible_storage(self):
        """Create a mock bible storage object when dependencies are not available"""
        class MockBibleStorage:
            def create_entry(self, *args, **kwargs):
                return {"id": 1, "content": {}, "type": "mock"}
                
            def get_entry(self, *args, **kwargs):
                return {"id": 1, "content": {}, "type": "mock"}
                
            def update_entry(self, *args, **kwargs):
                return {"id": 1, "content": {}, "type": "mock"}
                
            def delete_entry(self, *args, **kwargs):
                return True
                
            def list_entries(self, *args, **kwargs):
                return []
        
        return MockBibleStorage()
        
    def _create_mock_llm_wrapper(self):
        """Create a mock LLM wrapper object when dependencies are not available"""
        class MockLLMWrapper:
            def generate_text(self, prompt, *args, **kwargs):
                return {
                    "text": f"Mock response for '{prompt[:30]}...'",
                    "model": "mock-model",
                    "provider": "mock"
                }
        
        return MockLLMWrapper()
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """
        Get default prompts if none are loaded from file.
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            "bootstrap_bibles": "Create character, plot, setting and theme details for the following log line: {{log_line}}",
            "scene_outline": "Create a scene outline for the following log line: {{log_line}} with these characters: {{characters}}",
            "dialogue_generation": "Write realistic dialogue for a scene where: {{scene_description}}"
        }
    
    def generate_script_from_logline(self, log_line: str) -> Dict[str, Any]:
        """
        Generate a complete script scene from a log line.
        
        Args:
            log_line: The log line to generate from
            
        Returns:
            Dictionary containing the generated script and metadata
        """
        logger.info(f"Generating script from log line: {log_line}")
        
        # Step 1: Bootstrap bible entries
        bibles = self._bootstrap_bibles(log_line)
        
        # Step 2: Generate scene outline
        scene_outline = self._generate_scene_outline(log_line, bibles)
        
        # Step 3: Generate dialogue
        dialogue = self._generate_dialogue(scene_outline, bibles)
        
        # Combine results
        script = {
            "log_line": log_line,
            "bibles": bibles,
            "scene_outline": scene_outline,
            "dialogue": dialogue,
            "full_script": self._format_full_script(scene_outline, dialogue)
        }
        
        return script
    
    def _bootstrap_bibles(self, log_line: str) -> Dict[str, Any]:
        """
        Generate initial bible entries from a log line.
        
        Args:
            log_line: The log line to bootstrap from
            
        Returns:
            Dictionary of bible entries
        """
        logger.info("Bootstrapping bible entries")
        
        # Prepare prompt
        prompt = self.prompts.get("bootstrap_bibles", "").replace("{{log_line}}", log_line)
        
        # Get response from LLM
        response = self.llm_wrapper.generate_text(prompt)
        
        # Parse response into bible entries
        try:
            # Try to parse as JSON first
            if response["text"].strip().startswith("{") and response["text"].strip().endswith("}"):
                entries = json.loads(response["text"])
            else:
                # Fall back to simple parsing
                entries = self._simple_parse_entries(response["text"])
        except json.JSONDecodeError:
            entries = self._simple_parse_entries(response["text"])
        
        # Store entries in database
        stored_entries = {}
        for entry_type, content in entries.items():
            if entry_type == "character":
                entry = self.bible_storage.create_entry("character", content)
                stored_entries["character"] = entry
            elif entry_type == "plot":
                entry = self.bible_storage.create_entry("plot", content)
                stored_entries["plot"] = entry
            elif entry_type == "setting":
                entry = self.bible_storage.create_entry("setting", content)
                stored_entries["setting"] = entry
            elif entry_type == "theme":
                entry = self.bible_storage.create_entry("theme", content)
                stored_entries["theme"] = entry
        
        return stored_entries
    
    def _simple_parse_entries(self, text: str) -> Dict[str, Any]:
        """
        Simple parsing of LLM output into bible entries.
        
        Args:
            text: The LLM output text
            
        Returns:
            Dictionary of bible entries
        """
        entries = {
            "character": {},
            "plot": {},
            "setting": {},
            "theme": {}
        }
        
        # Very basic parsing - in real implementation, this would be more robust
        current_section = None
        current_content = []
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "CHARACTER" in line.upper():
                if current_section and current_content:
                    entries[current_section] = "\n".join(current_content)
                current_section = "character"
                current_content = []
            elif "PLOT" in line.upper():
                if current_section and current_content:
                    entries[current_section] = "\n".join(current_content)
                current_section = "plot"
                current_content = []
            elif "SETTING" in line.upper():
                if current_section and current_content:
                    entries[current_section] = "\n".join(current_content)
                current_section = "setting"
                current_content = []
            elif "THEME" in line.upper():
                if current_section and current_content:
                    entries[current_section] = "\n".join(current_content)
                current_section = "theme"
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            entries[current_section] = "\n".join(current_content)
            
        return entries
    
    def _generate_scene_outline(self, log_line: str, bibles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a scene outline from a log line and bible entries.
        
        Args:
            log_line: The log line
            bibles: Bible entries
            
        Returns:
            Scene outline dictionary
        """
        logger.info("Generating scene outline")
        
        # Extract character information
        character_info = ""
        if "character" in bibles:
            try:
                if isinstance(bibles["character"]["content"], dict):
                    character_info = json.dumps(bibles["character"]["content"])
                else:
                    character_info = str(bibles["character"]["content"])
            except:
                character_info = str(bibles["character"])
        
        # Prepare prompt
        prompt = self.prompts.get("scene_outline", "")
        prompt = prompt.replace("{{log_line}}", log_line)
        prompt = prompt.replace("{{characters}}", character_info)
        
        # Get response from LLM
        response = self.llm_wrapper.generate_text(prompt)
        
        # Parse response
        try:
            if response["text"].strip().startswith("{") and response["text"].strip().endswith("}"):
                outline = json.loads(response["text"])
            else:
                outline = {"description": response["text"]}
        except json.JSONDecodeError:
            outline = {"description": response["text"]}
        
        return outline
    
    def _generate_dialogue(self, scene_outline: Dict[str, Any], bibles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dialogue for a scene.
        
        Args:
            scene_outline: Scene outline
            bibles: Bible entries
            
        Returns:
            Dialogue dictionary
        """
        logger.info("Generating dialogue")
        
        # Extract scene description
        scene_description = ""
        if isinstance(scene_outline, dict):
            if "description" in scene_outline:
                scene_description = scene_outline["description"]
            else:
                scene_description = json.dumps(scene_outline)
        else:
            scene_description = str(scene_outline)
        
        # Prepare prompt
        prompt = self.prompts.get("dialogue_generation", "")
        prompt = prompt.replace("{{scene_description}}", scene_description)
        
        # Get response from LLM
        response = self.llm_wrapper.generate_text(prompt)
        
        # Parse response
        try:
            if response["text"].strip().startswith("{") and response["text"].strip().endswith("}"):
                dialogue = json.loads(response["text"])
            else:
                # Try to parse as a screenplay format
                dialogue = self._parse_screenplay_format(response["text"])
        except json.JSONDecodeError:
            dialogue = self._parse_screenplay_format(response["text"])
        
        return dialogue
    
    def _parse_screenplay_format(self, text: str) -> Dict[str, Any]:
        """
        Parse screenplay format text.
        
        Args:
            text: The screenplay text
            
        Returns:
            Structured dialogue
        """
        lines = []
        current_speaker = None
        current_dialogue = []
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a character name (usually in all caps)
            if line.isupper() and len(line) < 30:
                # Save previous speaker's dialogue
                if current_speaker and current_dialogue:
                    lines.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_dialogue)
                    })
                    
                # Start new speaker
                current_speaker = line
                current_dialogue = []
            elif current_speaker and not line.startswith("(") and not line.upper().startswith("INT.") and not line.upper().startswith("EXT."):
                # This is dialogue for the current speaker
                current_dialogue.append(line)
        
        # Add the last speaker
        if current_speaker and current_dialogue:
            lines.append({
                "speaker": current_speaker,
                "text": " ".join(current_dialogue)
            })
            
        return {"lines": lines}
    
    def _format_full_script(self, scene_outline: Dict[str, Any], dialogue: Dict[str, Any]) -> str:
        """
        Format the full script.
        
        Args:
            scene_outline: Scene outline
            dialogue: Dialogue
            
        Returns:
            Formatted script as string
        """
        # Get scene heading and description
        scene_heading = ""
        scene_description = ""
        
        if isinstance(scene_outline, dict):
            if "location" in scene_outline:
                scene_heading = scene_outline["location"]
            if "description" in scene_outline:
                scene_description = scene_outline["description"]
        
        # Format script
        script_lines = []
        
        # Add scene heading
        if scene_heading:
            script_lines.append(scene_heading.upper())
        else:
            script_lines.append("INT. UNKNOWN LOCATION - DAY")
            
        script_lines.append("")
        
        # Add scene description
        if scene_description:
            script_lines.append(scene_description)
            script_lines.append("")
        
        # Add dialogue
        if isinstance(dialogue, dict) and "lines" in dialogue:
            for line in dialogue["lines"]:
                if "speaker" in line:
                    script_lines.append(line["speaker"])
                    
                if "text" in line:
                    script_lines.append(f"    {line['text']}")
                    
                script_lines.append("")
        
        return "\n".join(script_lines)


def create_script_generator(
    config_path: Optional[str] = None, 
    db_path: Optional[str] = None
) -> ScriptGenerator:
    """
    Create a script generator with default components.
    
    Args:
        config_path: Path to LLM config
        db_path: Path to database
        
    Returns:
        Configured ScriptGenerator instance
    """
    if not DEPENDENCIES_AVAILABLE:
        return ScriptGenerator()
        
    # Create components
    bible_storage = BibleStorage(db_path=db_path)
    llm_wrapper = LLMWrapper(config_path=config_path)
    
    # Create generator
    return ScriptGenerator(
        bible_storage=bible_storage,
        llm_wrapper=llm_wrapper
    ) 