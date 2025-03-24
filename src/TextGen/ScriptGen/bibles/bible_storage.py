import json
import datetime
import difflib
from typing import Dict, List, Optional, Union, Any

from ..core.models import (
    Bible, CharacterBible, PlotBible, SettingBible, ThemeBible,
    get_engine, initialize_database
)

class BibleStorage:
    """
    Centralized system to create, store, update, and retrieve "bibles" 
    (Character, Plot, Setting, Theme) as structured data.
    """
    
    def __init__(self, db_path='script_gen.db'):
        self.engine = get_engine(db_path)
        self.Session = initialize_database(self.engine)
        
    def _get_bible_class(self, bible_type: str):
        """Get the appropriate Bible class based on type"""
        type_map = {
            'character': CharacterBible,
            'plot': PlotBible,
            'setting': SettingBible,
            'theme': ThemeBible
        }
        
        if bible_type.lower() not in type_map:
            raise ValueError(f"Invalid bible type: {bible_type}. Must be one of {list(type_map.keys())}")
            
        return type_map[bible_type.lower()]
    
    def create_bible(self, bible_type: str, content: Dict[str, Any]) -> int:
        """
        Create a new bible entry.
        
        Args:
            bible_type: Type of bible ('character', 'plot', 'setting', 'theme')
            content: Dictionary containing bible content
            
        Returns:
            ID of the newly created bible entry
        """
        bible_class = self._get_bible_class(bible_type)
        
        with self.Session() as session:
            bible = bible_class(content=content)
            session.add(bible)
            session.commit()
            return bible.id
    
    def get_bible(self, bible_id: int) -> Dict[str, Any]:
        """
        Retrieve a specific bible entry by ID.
        
        Args:
            bible_id: ID of the bible entry to retrieve
            
        Returns:
            Dictionary containing bible data including content and metadata
        """
        with self.Session() as session:
            bible = session.query(Bible).filter(Bible.id == bible_id).first()
            
            if not bible:
                raise ValueError(f"Bible with ID {bible_id} not found")
                
            return {
                'id': bible.id,
                'type': bible.type,
                'content': bible.content,
                'created_at': bible.created_at.isoformat(),
                'updated_at': bible.updated_at.isoformat(),
                'version_history': json.loads(bible.version_history)
            }
    
    def get_all_bibles(self, bible_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all bible entries, optionally filtered by type.
        
        Args:
            bible_type: Optional filter for bible type
            
        Returns:
            List of dictionaries containing bible data
        """
        with self.Session() as session:
            query = session.query(Bible)
            
            if bible_type:
                bible_class = self._get_bible_class(bible_type)
                query = query.filter(Bible.type == bible_type.lower())
            
            bibles = query.all()
            
            return [
                {
                    'id': bible.id,
                    'type': bible.type,
                    'content': bible.content,
                    'created_at': bible.created_at.isoformat(),
                    'updated_at': bible.updated_at.isoformat()
                }
                for bible in bibles
            ]
    
    def update_bible(self, bible_id: int, content: Dict[str, Any], store_version: bool = True) -> Dict[str, Any]:
        """
        Update an existing bible entry.
        
        Args:
            bible_id: ID of the bible to update
            content: New content for the bible
            store_version: Whether to store version history
            
        Returns:
            Updated bible data
        """
        with self.Session() as session:
            bible = session.query(Bible).filter(Bible.id == bible_id).first()
            
            if not bible:
                raise ValueError(f"Bible with ID {bible_id} not found")
            
            # Create version history entry if enabled
            if store_version:
                old_content = json.dumps(bible.content, sort_keys=True, indent=2)
                new_content = json.dumps(content, sort_keys=True, indent=2)
                
                diff = list(difflib.unified_diff(
                    old_content.splitlines(),
                    new_content.splitlines(),
                    n=0
                ))
                
                version_history = json.loads(bible.version_history)
                version_history.append({
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'diff': diff
                })
                
                bible.version_history = json.dumps(version_history)
            
            # Update content
            bible.content = content
            session.commit()
            
            return {
                'id': bible.id,
                'type': bible.type,
                'content': bible.content,
                'created_at': bible.created_at.isoformat(),
                'updated_at': bible.updated_at.isoformat(),
                'version_history': json.loads(bible.version_history)
            }
    
    def delete_bible(self, bible_id: int) -> bool:
        """
        Delete a bible entry by ID.
        
        Args:
            bible_id: ID of the bible to delete
            
        Returns:
            True if deletion was successful
        """
        with self.Session() as session:
            bible = session.query(Bible).filter(Bible.id == bible_id).first()
            
            if not bible:
                raise ValueError(f"Bible with ID {bible_id} not found")
                
            session.delete(bible)
            session.commit()
            return True
    
    def revert_to_version(self, bible_id: int, version_index: int) -> Dict[str, Any]:
        """
        Revert a bible to a previous version.
        
        Args:
            bible_id: ID of the bible to revert
            version_index: Index of the version to revert to
            
        Returns:
            Updated bible data
        """
        with self.Session() as session:
            bible = session.query(Bible).filter(Bible.id == bible_id).first()
            
            if not bible:
                raise ValueError(f"Bible with ID {bible_id} not found")
                
            version_history = json.loads(bible.version_history)
            
            if version_index < 0 or version_index >= len(version_history):
                raise ValueError(f"Invalid version index: {version_index}")
            
            # Start with the original content and apply diffs up to the target version
            current_content = bible.content
            
            # Create a copy of the current content to serve as a snapshot
            snapshot = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'content': current_content
            }
            
            # Reset version history to include only versions up to the specified index
            bible.version_history = json.dumps(version_history[:version_index + 1])
            
            # Commit changes
            session.commit()
            
            return {
                'id': bible.id,
                'type': bible.type,
                'content': bible.content,
                'created_at': bible.created_at.isoformat(),
                'updated_at': bible.updated_at.isoformat(),
                'version_history': json.loads(bible.version_history)
            } 