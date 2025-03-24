# ScriptGen Orchestrator Module

The Orchestrator module serves as the central integration layer for the ScriptGen system, providing workflow management, optimization, and a unified interface for script generation.

## Components

### Workflow Controller (`workflow_controller.py`)

The `WorkflowController` class is the core component of the Orchestrator module. It:

- Coordinates the script generation pipeline from start to finish
- Manages data flow between different components (bibles, generators, agents)
- Provides parallel processing and optimized execution
- Supports different generation scopes (single scene, multi-scene, full script)

```python
from ScriptGen.orchestrator.workflow_controller import WorkflowController, GenerationConfig

controller = WorkflowController()
result = controller.generate_script(
    log_line="A detective must solve a case that hits too close to home.",
    config=GenerationConfig(scene_count=3)
)
```

### LLM Optimizer (`optimizer.py`)

The `LLMOptimizer` class provides optimization strategies for LLM usage:

- **Caching**: Stores and reuses LLM responses
- **Batching**: Groups similar LLM calls to reduce overhead
- **Token Management**: Optimizes token usage and tracks costs

```python
from ScriptGen.orchestrator.optimizer import LLMOptimizer
from ScriptGen.llm.llm_wrapper import LLMWrapper

llm_wrapper = LLMWrapper()
optimizer = LLMOptimizer(llm_wrapper)

# Single optimized call
response = optimizer.call_llm(prompt="Generate a character description")

# Batch calls
optimizer.start_batch()
result1 = optimizer.call_llm(prompt="Generate dialogue for character 1")
result2 = optimizer.call_llm(prompt="Generate dialogue for character 2")
optimizer.end_batch()
```

## Configuration

### GenerationConfig

The `GenerationConfig` class provides configuration options for script generation:

```python
from ScriptGen.orchestrator.workflow_controller import GenerationConfig, GenerationScope

config = GenerationConfig(
    scope=GenerationScope.MULTI_SCENE,  # SINGLE_SCENE, MULTI_SCENE, FULL_SCRIPT
    character_count=3,                  # Number of characters to generate
    scene_count=3,                      # Number of scenes to generate
    use_existing_bibles=False,          # Generate new bibles or use existing
    auto_rewrite=True,                  # Apply automatic rewrites
    rewrite_iterations=2,               # Number of rewrite iterations
    parallel_generation=True,           # Enable parallel generation
    optimize_llm_calls=True             # Enable LLM optimization
)
```

### Optimization Settings

LLM optimization can be configured through the `OptimizerConfig` class:

```python
from ScriptGen.orchestrator.optimizer import OptimizerConfig

optimizer_config = OptimizerConfig(
    enable_caching=True,           # Enable response caching
    cache_expire_hours=24,         # Cache expiration time
    enable_batching=True,          # Enable call batching
    batch_timeout_ms=500,          # Batch collection timeout
    enable_deduplication=True,     # Enable prompt deduplication
    dedup_window_seconds=30        # Deduplication time window
)
```

## Integration Examples

### Basic Script Generation

```python
from ScriptGen.orchestrator.workflow_controller import (
    WorkflowController, GenerationConfig, GenerationScope
)

controller = WorkflowController()

config = GenerationConfig(
    scope=GenerationScope.SINGLE_SCENE,
    character_count=2
)

result = controller.generate_script(
    log_line="Two strangers meet on a train and discover they have the same target.",
    config=config
)

print(result['scenes'][0]['integrated_scene'])
```

### Using Existing Bibles

```python
from ScriptGen.bibles.bible_storage import BibleStorage

# Initialize components
bible_storage = BibleStorage()
controller = WorkflowController(bible_storage=bible_storage)

# Create or retrieve bible entries
character_ids = []
character_ids.append(bible_storage.create_entry('character', {
    'name': 'Jack',
    'backstory': 'Former spy now working as a private investigator',
    'traits': ['resourceful', 'cynical', 'determined']
}))

# Generate script with existing bibles
result = controller.generate_script(
    log_line="A PI is hired to find a missing heirloom.",
    config=GenerationConfig(use_existing_bibles=True),
    existing_bibles={'character': character_ids}
)
```

### Multi-Scene Generation with Optimization

```python
from ScriptGen.llm.llm_wrapper import LLMWrapper
from ScriptGen.orchestrator.optimizer import LLMOptimizer, OptimizerConfig

# Initialize components
llm_wrapper = LLMWrapper()
optimizer = LLMOptimizer(
    llm_wrapper, 
    config=OptimizerConfig(enable_caching=True, enable_batching=True)
)
controller = WorkflowController(llm_wrapper=llm_wrapper, llm_optimizer=optimizer)

# Configure generation
config = GenerationConfig(
    scope=GenerationScope.MULTI_SCENE,
    scene_count=3,
    parallel_generation=True,
    optimize_llm_calls=True
)

# Generate script
result = controller.generate_script(
    log_line="A family vacation turns into a survival adventure when their plane crashes.",
    config=config
)

# Print scene summaries
for i, scene in enumerate(result['scenes']):
    print(f"Scene {i+1}: {scene['outline']['description']}")
```

## Advanced Usage

### Custom Pipeline

You can create custom pipelines by directly interacting with individual components:

```python
# Manual pipeline example
bible_result = controller.bootstrap_bibles(log_line="A heist gone wrong.", character_count=4)
scenes = []

for i in range(3):
    scene_outline = controller.generate_scene_outline(
        log_line="A heist gone wrong.",
        bibles=bible_result,
        scene_number=i
    )
    scene = controller.generate_scene(
        outline=scene_outline,
        bibles=bible_result
    )
    scenes.append(scene)
    
    # Apply rewrites if needed
    if i == 1:  # Apply rewrites only to the second scene
        rewrite_result = controller.rewrite_scene(
            scene=scene,
            bibles=bible_result,
            critique="Make the dialogue more tense."
        )
        scenes[i] = rewrite_result

# Format final result
final_script = controller.format_script(scenes, bible_result)
```

### Performance Monitoring

```python
# Get optimization statistics
if controller.llm_optimizer:
    stats = controller.llm_optimizer.get_stats()
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Total calls: {stats['total_calls']}")
    print(f"Estimated savings: ${stats['estimated_cost_savings']:.2f}")
    print(f"Tokens saved: {stats['tokens_saved']}")
```

## Error Handling

The Workflow Controller provides robust error handling:

```python
try:
    result = controller.generate_script(log_line="A suspenseful thriller.")
except Exception as e:
    # Handle specific error types
    if "Bible generation failed" in str(e):
        # Try with default bibles
        result = controller.generate_script(
            log_line="A suspenseful thriller.",
            config=GenerationConfig(use_existing_bibles=False)
        )
    elif "LLM service unavailable" in str(e):
        # Try with alternate LLM provider
        controller.llm_wrapper.switch_provider("alternate_provider")
        result = controller.generate_script(log_line="A suspenseful thriller.")
    else:
        # Generic error handling
        print(f"Script generation failed: {e}")
``` 