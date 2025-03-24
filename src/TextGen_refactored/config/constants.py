"""
Constants and default values used throughout the DramaTron system.
"""
from typing import Dict, Any

# Default seed for reproducibility
DEFAULT_SEED = 1

# Sampling configuration
SAMPLING = {
    'PROB': 0.9,
    'PROB_MISTRAL': 0.7,
    'TEMP': 1.0
}

# Sample length limits
SAMPLE_LENGTH = {
    'TITLE': 64,
    'PLACE': 512,
    'DEFAULT': 1024,
    'SONNET': 2048
}

# Maximum paragraph lengths
MAX_PARAGRAPH_LENGTH = {
    'CHARACTERS': 1024,
    'SCENES': 1024,
    'SCENES_MISTRAL': 700,
    'DEFAULT': 3000
}

# Retry and timeout settings
MAX_RETRIES = 10
TIMEOUT = 120.0
MAX_NUM_REPETITIONS = 4
MAX_NUM_ATTEMPTS_GET_OUT_OF_LOOP = 4

# Markers for text generation
MARKERS = {
    'END': '**END**',
    'STOP': '\n',
    'CHARACTER': '**Character:** ',
    'DESCRIPTION': '**Description:** ',
    'SCENES': '**Scenes:**',
    'DIALOG': '**Dialog:**',
    'LOGLINE': '**Logline:** '
}

# Elements for template generation
ELEMENTS = {
    'TITLE': 'Title: ',
    'CHARACTERS': 'Characters: ',
    'DESCRIPTION': 'Description: ',
    'PLACE': 'Place: ',
    'PLOT': 'Plot element: ',
    'PREVIOUS': 'Previous beat: ',
    'SUMMARY': 'Summary: ',
    'BEAT': 'Beat: ',
    'LOGLINE': 'Logline: '
}

# Default configuration dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    'language_api_name': 'OpenAI',
    'max_retries': MAX_RETRIES,
    'sample_length': SAMPLE_LENGTH['DEFAULT'],
    'max_paragraph_length': MAX_PARAGRAPH_LENGTH['DEFAULT'],
    'max_paragraph_length_characters': MAX_PARAGRAPH_LENGTH['CHARACTERS'],
    'max_paragraph_length_scenes': MAX_PARAGRAPH_LENGTH['SCENES'],
    'sampling': {
        'prob': SAMPLING['PROB'],
        'temp': SAMPLING['TEMP']
    },
    'prefixes': {},
    'file_dir': None
} 