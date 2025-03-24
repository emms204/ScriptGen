# ScriptGen System

A powerful script generation system that uses LLMs to create script scenes from log lines, manages character/plot/setting bibles, and provides a foundation for advanced story development.

## Project Structure

The project is organized into modules focusing on specific aspects of script generation:

```
src/TextGen/ScriptGen/
├── bibles/                    # Bible storage and management
├── core/                      # Core components and models
├── generators/                # Script generation engines
├── llm/                       # LLM API wrappers
├── tests/                     # Unit and integration tests
└── utils/                     # Utility functions and helpers
```

See `src/TextGen/ScriptGen/STRUCTURE.md` for detailed information about the project organization and future development phases.

## Setup

1. Clone the repository
2. Install the required dependencies:

```bash
pip install sqlalchemy openai requests
```

3. Configure the LLM API keys:

Create a configuration file at `src/TextGen/ScriptGen/config/models.json` with the following structure:

```json
{
  "default_model": "gpt-3.5-turbo",
  "models": {
    "openai": {
      "api_key": "your-openai-api-key",
      "available_models": ["gpt-3.5-turbo", "gpt-4"]
    }
  }
}
```

Alternatively, set the `OPENAI_API_KEY` environment variable.

## Usage

### Basic Script Generation

```python
from TextGen.ScriptGen.generators import create_script_generator

# Create a script generator
generator = create_script_generator()

# Generate a script from a log line
script = generator.generate_script_from_logline(
    "A detective with amnesia must solve a murder that he might have committed."
)

# Print the generated script
print(script['full_script'])
```

### Using the Example Script

Run the example script to generate a sample script:

```bash
cd src
python -m TextGen.ScriptGen.examples.example "A detective with amnesia must solve a murder that he might have committed."
```

## Documentation

For detailed documentation on each module, see:

- `src/TextGen/ScriptGen/IMPLEMENTATION_SUMMARY.md` - Summary of implemented features
- `src/TextGen/ScriptGen/STRUCTURE.md` - Project structure and future development

## Testing

Run the tests:

```bash
cd src
python -m unittest discover -s TextGen/ScriptGen/tests
```