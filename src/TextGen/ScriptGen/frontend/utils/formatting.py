"""
Text formatting utilities for the Streamlit frontend.

This module provides functions for formatting text, especially for screenplay format.
"""

import re
import html
import difflib
from typing import Dict, List, Any, Optional, Tuple


def format_screenplay_html(text: str) -> str:
    """
    Format screenplay text to HTML with proper styling.
    
    Args:
        text: Raw screenplay text
        
    Returns:
        HTML formatted text
    """
    if not text:
        return ""
    
    # Escape HTML
    text = html.escape(text)
    
    # Scene headings (INT./EXT.)
    text = re.sub(
        r'^(INT\.|EXT\.|INT\./EXT\.|I/E)(.+)$',
        r'<div class="scene-heading">\1\2</div>',
        text,
        flags=re.MULTILINE
    )
    
    # Character names (ALL CAPS followed by dialogue)
    text = re.sub(
        r'^([A-Z][A-Z\s]+)(\(.*\))?$',
        r'<div class="character-name">\1\2</div>',
        text,
        flags=re.MULTILINE
    )
    
    # Parentheticals
    text = re.sub(
        r'^\((.*)\)$',
        r'<div class="parenthetical">(\1)</div>',
        text,
        flags=re.MULTILINE
    )
    
    # Convert line breaks to <br>
    text = text.replace('\n', '<br>')
    
    return f"""
    <style>
        .screenplay {{
            font-family: 'Courier New', monospace;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.5;
        }}
        .scene-heading {{
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .character-name {{
            margin-top: 15px;
            margin-bottom: 0;
            margin-left: 150px;
        }}
        .parenthetical {{
            margin-top: 0;
            margin-bottom: 0;
            margin-left: 120px;
        }}
        .dialogue {{
            margin-top: 0;
            margin-left: 100px;
            margin-right: 100px;
            margin-bottom: 10px;
        }}
    </style>
    <div class="screenplay">{text}</div>
    """


def format_diff_html(old_text: str, new_text: str) -> str:
    """
    Generate HTML for diff between old and new text.
    
    Args:
        old_text: Original text
        new_text: New text
        
    Returns:
        HTML formatted diff
    """
    if not old_text and not new_text:
        return ""
    
    # Generate diff
    diff = difflib.HtmlDiff(tabsize=4)
    html_diff = diff.make_file(
        old_text.splitlines(),
        new_text.splitlines(),
        fromdesc="Original",
        todesc="Modified"
    )
    
    # Add custom styling
    html_diff = html_diff.replace(
        '<style type="text/css">',
        """<style type="text/css">
        table.diff {
            font-family: 'Courier New', monospace;
            border-collapse: collapse;
            width: 100%;
        }
        .diff_header {
            background-color: #f3f3f3;
            text-align: center;
            width: 20px;
        }
        td.diff_header {
            text-align: right;
        }
        .diff_next {
            background-color: #f3f3f3;
            text-align: center;
            width: 20px;
        }
        .diff_add {
            background-color: #ddffdd;
            width: 50%;
        }
        .diff_chg {
            background-color: #ffffcc;
            width: 50%;
        }
        .diff_sub {
            background-color: #ffdddd;
            width: 50%;
        }
        """
    )
    
    return html_diff


def format_bible_html(bible_entry: Dict[str, Any]) -> str:
    """
    Format bible entry as HTML.
    
    Args:
        bible_entry: Bible entry data
        
    Returns:
        HTML formatted bible entry
    """
    if not bible_entry:
        return ""
    
    entry_type = bible_entry.get('type', 'unknown')
    content = bible_entry.get('content', {})
    
    if entry_type == 'character':
        return format_character_bible_html(content)
    elif entry_type == 'plot':
        return format_plot_bible_html(content)
    elif entry_type == 'setting':
        return format_setting_bible_html(content)
    elif entry_type == 'theme':
        return format_theme_bible_html(content)
    else:
        return format_generic_bible_html(content)


def format_character_bible_html(content: Dict[str, Any]) -> str:
    """Format character bible as HTML."""
    html_parts = [f"<h3>{content.get('name', 'Unnamed Character')}</h3>"]
    
    if 'backstory' in content:
        html_parts.append(f"<p><strong>Backstory:</strong> {html.escape(content['backstory'])}</p>")
    
    if 'goals' in content:
        html_parts.append(f"<p><strong>Goals:</strong> {html.escape(content['goals'])}</p>")
    
    if 'fears' in content:
        html_parts.append(f"<p><strong>Fears:</strong> {html.escape(content['fears'])}</p>")
    
    if 'traits' in content and content['traits']:
        traits = content['traits']
        if isinstance(traits, list):
            traits_str = ', '.join(traits)
        else:
            traits_str = str(traits)
        html_parts.append(f"<p><strong>Traits:</strong> {html.escape(traits_str)}</p>")
    
    if 'relationships' in content and content['relationships']:
        html_parts.append("<p><strong>Relationships:</strong></p><ul>")
        for person, relationship in content['relationships'].items():
            html_parts.append(f"<li><strong>{html.escape(person)}:</strong> {html.escape(relationship)}</li>")
        html_parts.append("</ul>")
    
    return "\n".join(html_parts)


def format_plot_bible_html(content: Dict[str, Any]) -> str:
    """Format plot bible as HTML."""
    html_parts = [f"<h3>{content.get('title', 'Unnamed Plot')}</h3>"]
    
    if 'theme' in content:
        html_parts.append(f"<p><strong>Theme:</strong> {html.escape(content['theme'])}</p>")
    
    if 'main_conflict' in content:
        html_parts.append(f"<p><strong>Main Conflict:</strong> {html.escape(content['main_conflict'])}</p>")
    
    if 'act_structure' in content:
        html_parts.append(f"<p><strong>Structure:</strong> {html.escape(content['act_structure'])}</p>")
    
    if 'scenes' in content and content['scenes']:
        html_parts.append("<p><strong>Scenes:</strong></p><ul>")
        for scene in content['scenes']:
            description = scene.get('description', 'No description')
            html_parts.append(f"<li>{html.escape(description)}</li>")
        html_parts.append("</ul>")
    
    return "\n".join(html_parts)


def format_setting_bible_html(content: Dict[str, Any]) -> str:
    """Format setting bible as HTML."""
    html_parts = [f"<h3>{content.get('name', 'Unnamed Setting')}</h3>"]
    
    if 'description' in content:
        html_parts.append(f"<p><strong>Description:</strong> {html.escape(content['description'])}</p>")
    
    if 'time_period' in content:
        html_parts.append(f"<p><strong>Time Period:</strong> {html.escape(content['time_period'])}</p>")
    
    if 'mood' in content:
        html_parts.append(f"<p><strong>Mood:</strong> {html.escape(content['mood'])}</p>")
    
    if 'locations' in content and content['locations']:
        html_parts.append("<p><strong>Locations:</strong></p><ul>")
        for location in content['locations']:
            if isinstance(location, dict):
                name = location.get('name', 'Unnamed')
                desc = location.get('description', '')
                html_parts.append(f"<li><strong>{html.escape(name)}:</strong> {html.escape(desc)}</li>")
            else:
                html_parts.append(f"<li>{html.escape(str(location))}</li>")
        html_parts.append("</ul>")
    
    return "\n".join(html_parts)


def format_theme_bible_html(content: Dict[str, Any]) -> str:
    """Format theme bible as HTML."""
    html_parts = [f"<h3>{content.get('name', 'Unnamed Theme')}</h3>"]
    
    if 'description' in content:
        html_parts.append(f"<p><strong>Description:</strong> {html.escape(content['description'])}</p>")
    
    if 'symbols' in content and content['symbols']:
        symbols = content['symbols']
        if isinstance(symbols, list):
            symbols_str = ', '.join(symbols)
        else:
            symbols_str = str(symbols)
        html_parts.append(f"<p><strong>Symbols:</strong> {html.escape(symbols_str)}</p>")
    
    if 'character_arcs' in content and content['character_arcs']:
        html_parts.append("<p><strong>Character Arcs:</strong></p><ul>")
        for char, arc in content['character_arcs'].items():
            html_parts.append(f"<li><strong>{html.escape(char)}:</strong> {html.escape(arc)}</li>")
        html_parts.append("</ul>")
    
    return "\n".join(html_parts)


def format_generic_bible_html(content: Dict[str, Any]) -> str:
    """Format generic bible entry as HTML."""
    html_parts = ["<h3>Bible Entry</h3>"]
    
    for key, value in content.items():
        if isinstance(value, dict):
            html_parts.append(f"<p><strong>{html.escape(key.title())}:</strong></p><ul>")
            for subkey, subvalue in value.items():
                html_parts.append(f"<li><strong>{html.escape(subkey)}:</strong> {html.escape(str(subvalue))}</li>")
            html_parts.append("</ul>")
        elif isinstance(value, list):
            html_parts.append(f"<p><strong>{html.escape(key.title())}:</strong></p><ul>")
            for item in value:
                html_parts.append(f"<li>{html.escape(str(item))}</li>")
            html_parts.append("</ul>")
        else:
            html_parts.append(f"<p><strong>{html.escape(key.title())}:</strong> {html.escape(str(value))}</p>")
    
    return "\n".join(html_parts)


def format_critique_html(critique: Dict[str, Any]) -> str:
    """
    Format dramaturge critique as HTML.
    
    Args:
        critique: Critique data
        
    Returns:
        HTML formatted critique
    """
    if not critique:
        return ""
    
    html_parts = ["<h3>Scene Critique</h3>"]
    
    if 'overall_assessment' in critique:
        html_parts.append(f"<p><strong>Overall:</strong> {html.escape(critique['overall_assessment'])}</p>")
    
    sections = ['pacing', 'tension', 'subtext', 'character_authenticity', 'plot_advancement']
    
    for section in sections:
        if section in critique:
            section_data = critique[section]
            html_parts.append(f"<h4>{section.replace('_', ' ').title()}</h4>")
            
            if 'issues' in section_data and section_data['issues']:
                html_parts.append("<p><strong>Issues:</strong></p><ul>")
                for issue in section_data['issues']:
                    html_parts.append(f"<li>{html.escape(issue)}</li>")
                html_parts.append("</ul>")
            
            if 'suggestions' in section_data and section_data['suggestions']:
                html_parts.append("<p><strong>Suggestions:</strong></p><ul>")
                for suggestion in section_data['suggestions']:
                    html_parts.append(f"<li>{html.escape(suggestion)}</li>")
                html_parts.append("</ul>")
    
    if 'suggested_rewrites' in critique and critique['suggested_rewrites']:
        html_parts.append("<h4>Suggested Rewrites</h4><ul>")
        for rewrite in critique['suggested_rewrites']:
            html_parts.append(f"<li>{html.escape(rewrite)}</li>")
        html_parts.append("</ul>")
    
    return "\n".join(html_parts) 