# Phase 3: Streamlit-Based Frontend Interface

This document outlines the redesigned implementation plan for Phase 3 of the ScriptGen project, focusing on a Streamlit-based frontend interface that connects with the Iterative Rewrite Engine.

## Overview

### Advantages of Using Streamlit

1. **Rapid Development**: Streamlit allows for fast UI development with minimal code
2. **Python Integration**: Seamless integration with existing Python backend code
3. **Interactive Components**: Built-in widgets for user interaction (dropdowns, sliders, etc.)
4. **State Management**: Simple session state management for app flow
5. **Deployment Flexibility**: Easy deployment options (local, cloud, containers)

## UI Components and Layout

### 1. Navigation and Project Management
- **Sidebar**: Project selection, creation, and navigation between views
- **Authentication**: (Optional) Login/register functionality
- **Project Dashboard**: View all projects with key metadata

### 2. LLM Selection Component
- Dropdown selector for LLM provider
- Parameter configuration (temperature, max tokens)
- Sample output display for selected model
- "Try Model" button to test with sample prompts

### 3. Script Generation View
- Log line input field with "Generate" button
- Character selection multi-select box (from existing bibles)
- Plot selection dropdown
- Progress indicator during generation
- Generated scene display with tabs for:
  - Outline
  - Character Dialogue
  - Integrated Scene

### 4. Bible Management View
- Bible type tabs (Character, Plot, Setting, Theme)
- Bible entries displayed as expandable cards
- Add/Edit/Delete bible entry forms
- Relationship visualization for character bibles

### 5. Script Editing View
- Text area with screenplay formatting
- Inline editing capabilities
- Scene version history with diff view
- Export options (Markdown, PDF)

### 6. Rewrite Control Panel
- "Critique Scene" button for dramaturge feedback
- "Rewrite" input field with instructions
- "Preview Changes" button for bible propagation preview
- "Apply Changes" confirmation button
- "Revert" option for undoing changes

## Screen Layout Diagram

```
+-----------------------------------------------+
| SCRIPTGEN HEADER                              |
+------------+----------------------------------+
| SIDEBAR    |  MAIN CONTENT AREA               |
|            |                                  |
| Project:   |  [Tab Navigation]                |
| [Dropdown] |  Generate | Bibles | Edit | Review|
|            |                                  |
| View:      |  [Content Changes Based on Tab]  |
| [Radio]    |                                  |
|            |  +------------------------------+ |
| LLM:       |  | Tab 1: Script Generation     | |
| [Dropdown] |  | - Log Line Input             | |
|            |  | - Character/Plot Selection   | |
| Params:    |  | - Generated Scene Display    | |
| [Sliders]  |  +------------------------------+ |
|            |                                  |
| Actions:   |  +------------------------------+ |
| [Buttons]  |  | Tab 2: Bible Management      | |
|            |  | - Bible Type Tabs            | |
|            |  | - Entry Cards                | |
|            |  | - Edit Forms                 | |
|            |  +------------------------------+ |
|            |                                  |
|            |  +------------------------------+ |
|            |  | Tab 3: Script Editing        | |
|            |  | - Formatted Text Area        | |
|            |  | - Rewrite Controls           | |
|            |  | - Version History            | |
|            |  +------------------------------+ |
|            |                                  |
|            |  +------------------------------+ |
|            |  | Tab 4: Review & Feedback     | |
|            |  | - Dramaturge Critique        | |
|            |  | - Change Preview             | |
|            |  | - Diff View                  | |
|            |  +------------------------------+ |
+------------+----------------------------------+
```

## Implementation Structure

### File Organization

```
frontend/
├── app.py                 # Main Streamlit application entry point
├── pages/                 # Streamlit multi-page app structure
│   ├── 01_script_gen.py   # Script generation page
│   ├── 02_bibles.py       # Bible management page
│   ├── 03_edit.py         # Script editing page
│   └── 04_review.py       # Review and feedback page
├── utils/                 # Utility functions
│   ├── api.py             # API client for backend communication
│   ├── formatting.py      # Text formatting utilities
│   ├── session.py         # Session state management
│   └── visualization.py   # Data visualization helpers
├── components/            # Custom Streamlit components
│   ├── bible_editor.py    # Bible editing component
│   ├── script_viewer.py   # Script viewing/editing component
│   ├── llm_selector.py    # LLM selection component
│   └── diff_viewer.py     # Diff visualization component
└── styles/                # CSS styling
    └── custom.css         # Custom styling
```

## Backend Integration

### API Client Design

The frontend will communicate with the backend through a simple API client that wraps calls to the Iterative Rewrite Engine:

```python
class ScriptGenAPI:
    def __init__(self, base_url=None):
        """Initialize API client - if no base_url, use direct function calls"""
        self.base_url = base_url
        self.bible_storage = BibleStorage()
        self.llm_wrapper = LLMWrapper()
        self.rewrite_controller = RewriteController(
            bible_storage=self.bible_storage,
            llm_wrapper=self.llm_wrapper
        )
    
    def get_available_llms(self):
        """Get list of available LLM providers and models"""
        # Either API call or direct function call
        
    def create_bible_entry(self, entry_type, content):
        """Create a new bible entry"""
        # Call to bible_storage.create_entry
        
    def generate_scene(self, log_line, character_ids, plot_id, llm_config):
        """Generate a new scene"""
        # Call to script_generator functions
        
    def critique_scene(self, scene_id, scene_content):
        """Get critique of a scene"""
        # Call to rewrite_controller.critique_scene
        
    def rewrite_scene(self, scene_id, scene_content, user_input):
        """Rewrite a scene based on user instructions"""
        # Call to rewrite_controller.rewrite_scene
        
    def propagate_change(self, change_request):
        """Preview bible changes"""
        # Call to rewrite_controller.propagate_change
        
    def apply_propagation(self, change_plan, approved_scene_ids):
        """Apply bible changes to scenes"""
        # Call to rewrite_controller.apply_propagation
```

## Key Workflows

### 1. Script Generation Workflow

1. User selects project from sidebar
2. User navigates to "Generate" tab
3. User inputs log line
4. User selects characters and plot from dropdowns
5. User selects LLM from sidebar
6. User clicks "Generate Scene" button
7. UI shows progress indicator during generation
8. Generated scene appears in tabs (outline, dialogue, integrated)
9. User can save scene or continue editing

### 2. Bible Management Workflow

1. User navigates to "Bibles" tab
2. User selects bible type (Character, Plot, etc.)
3. User views list of existing entries
4. User can:
   - Create new entry with form
   - Edit existing entry by clicking on it
   - Delete entry with confirmation
5. Changes automatically propagate to affected scenes (with preview)

### 3. Scene Rewriting Workflow

1. User navigates to "Edit" tab
2. User selects scene to edit
3. User can:
   - Edit scene directly in text area
   - Request critique via "Critique Scene" button
   - Input rewrite instructions and click "Rewrite"
   - View previous versions in history panel
4. Rewritten scene appears with diff highlighting changes
5. User can accept or reject changes

### 4. Bible Change Propagation Workflow

1. User edits a bible entry (e.g., character name change)
2. System detects change and shows "Preview Changes" button
3. User clicks to see affected scenes with diffs
4. User can select which scenes to update
5. User confirms changes with "Apply Changes" button
6. System updates bible and selected scenes

## Implementation Plan

### Week 1: Setup and Core Infrastructure
- Set up Streamlit project structure
- Implement session state management
- Create API client for backend communication
- Implement authentication (if needed)

### Week 2: Script Generation and Bible Management
- Develop script generation page
- Implement LLM selection component
- Create bible management interface
- Build bible editing forms

### Week 3: Editing and Rewriting
- Implement script editing component
- Create rewrite control panel
- Develop version history functionality
- Build diff visualization

### Week 4: Review, Feedback, and Refinement
- Implement dramaturge critique display
- Create change propagation preview
- Add export functionality
- Polish UI and improve UX

### Week 5: Testing and Deployment
- Unit and integration testing
- User acceptance testing
- Fix bugs and address feedback
- Prepare for deployment

## Development Notes

### Streamlit Best Practices
- Use `st.session_state` for persistent state
- Cache expensive operations with `@st.cache_data`
- Create modular components using functions
- Handle state updates with callbacks
- Use containers for layout organization

### Technical Considerations
- Consider Streamlit performance with large text processing
- Implement async operations for long-running tasks
- Use appropriate widgets for different input types
- Ensure consistent styling and UX 