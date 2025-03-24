# TextGen Refactored

This is a refactored version of the Dramatron TextGen module, providing a more structured, maintainable, and extensible approach to story generation.

## Features

- **Modular Architecture**: Clear separation of concerns with distinct modules for models, providers, generators, and utilities.
- **Multiple LLM Provider Support**: Easily switch between OpenAI, Anthropic, Google Gemini models, and open-source models via Groq.
- **Open Source Model Support**: Access to Llama, Mixtral, Mistral, DeepSeek, Gemma, and Qwen models through Groq API.
- **Enhanced Story Generation**: Support for standard and enhanced story generators, with toxicity filtering.
- **Modern Templating**: Jinja2-style templating for creating dynamic prompts.
- **Multi-Model Comparison**: Compare outputs from multiple LLMs side by side.
- **Streamlit Web Interface**: User-friendly UI for story generation and editing.
- **Type Hints**: Comprehensive type annotations for better IDE support and code validation.
- **Extensive Documentation**: Clear docstrings and module documentation.

## Directory Structure

```
TextGen_refactored/
├── __init__.py              # Main package exports
├── config/                  # Configuration management
│   ├── __init__.py
│   └── constants.py         # Default configuration values
├── models/                  # Data models
│   ├── __init__.py
│   └── story.py             # Story data structures
├── providers/               # LLM provider implementations
│   ├── __init__.py
│   ├── base.py
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   └── groq_provider.py     # Access to open-source models
├── generators/              # Story generation logic
│   ├── __init__.py
│   ├── base.py
│   └── enhanced_generator.py
├── prompts/                 # Prompt templates
│   ├── __init__.py
│   └── templates.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── rendering.py
│   └── text_processing.py
├── interface/               # Web interface
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── cli.py
│   ├── multi_model.py
│   ├── session_state.py
│   └── export.py
├── run_dramatron.py         # Script to launch UI
└── requirements.txt         # Project dependencies
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Dramatron.git
   cd Dramatron
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r src/TextGen_refactored/requirements.txt
   ```

### API Keys Setup

Set up the necessary API keys for the LLM providers you intend to use:

1. **Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GEMINI_API_KEY="your-gemini-api-key"
   export GROQ_API_KEY="your-groq-api-key"
   ```

2. **Using `.env` file**:
   Create a `.env` file in the `src/TextGen_refactored` directory:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   GEMINI_API_KEY=your-gemini-api-key
   GROQ_API_KEY=your-groq-api-key
   ```

## Running the Application

### Web Interface

Launch the Streamlit web interface:

```bash
cd src/TextGen_refactored
python run_dramatron.py
```

Or directly with Streamlit:

```bash
streamlit run src/TextGen_refactored/interface/streamlit_app.py
```

The web interface will be available at http://localhost:8501 by default.

### Basic Usage

1. Set your API keys in the sidebar
2. Choose a single model or compare multiple models
3. Enter your logline/storyline
4. Select generation scope (Complete Story, Single Scene, etc.)
5. Click "Generate" to create your story
6. Edit or regenerate parts of your story as needed
7. Export the finished story

### Multi-Model Comparison

1. In the sidebar, check "Compare Multiple Models"
2. Select the models you want to compare
3. Enter your logline/storyline
4. Click "Generate"
5. View the outputs from different models in tabs

### Programmatic Usage

```python
from TextGen_refactored import (
    ProviderFactory,
    StoryGenerator,
    render_story
)

# Create a provider using the factory
factory = ProviderFactory()
provider = factory.create_provider(
    provider_type="openai",  # or "anthropic", "gemini", "groq"
    model_name="gpt-4",
    api_key="your-api-key",
    config_sampling={"temp": 0.7, "prob": 0.9}
)

# Create a story generator
generator = StoryGenerator(provider)

# Generate a story
logline = "A brilliant mathematician discovers an equation that predicts future events"
story = generator.generate_story(logline)

# Render the story
formatted_story = render_story(story)
print(formatted_story)
```

## Using Open-Source Models

The system supports open-source models through the Groq API:

```python
from TextGen_refactored import GroqProvider, StoryGenerator
from TextGen_refactored.providers.base import ProviderConfig

# Configure Groq provider
config = ProviderConfig(
    model="llama3-70b",  # or "mixtral-8x7b", "mistral-7b", "deepseek-llm", etc.
    api_key="your-groq-api-key",
    default_sample_length=1000,
    config_sampling={"temp": 0.7, "prob": 0.9}
)

# Create provider
provider = GroqProvider(config)

# Create generator and generate story
generator = StoryGenerator(provider)
story = generator.generate_story("Your logline here")
```

## Testing

Run tests to ensure everything is working correctly:

```bash
cd src/TextGen_refactored
python -m pytest
```

For more detailed test results with coverage:

```bash
python -m pytest --cov=. --cov-report=term-missing
```

## Extension Points

The module is designed to be easily extended:

- **Adding a new provider**: Create a new class that inherits from `LanguageModelProvider` in the providers directory
- **Creating custom generators**: Extend the `StoryGenerator` class with specialized generation logic
- **Customizing prompts**: Modify or add templates in the `prompts` module
- **Adding UI components**: Extend the Streamlit interface in the `interface` directory

## Troubleshooting

Common issues and their solutions:

1. **API Key Issues**: Ensure your API keys are correctly set either as environment variables or in the UI
2. **Missing Dependencies**: Make sure all requirements are installed with `pip install -r requirements.txt`
3. **UI Not Loading**: Check that Streamlit is installed and running on the correct port
4. **Model Errors**: Some models have limitations on input/output length - try adjusting the max tokens parameter

## License

[Include license information here]

## Contributors

[List contributors or link to contributors section]

## Acknowledgments

- Dramatron was developed by Piotr Mirowski and Kory W. Mathewson, with additional contributions by Juliette Love and Jaylen Pittman, and is based on a prototype by Richard Evans.
- This refactored version improves upon the original with a more maintainable and extensible architecture. 