# API Reference

Complete API documentation for Prompt-Sandbox.

---

## Configuration Module (`src/config/`)

### `PromptConfig`

Pydantic model for prompt configuration.

**Fields**:
```python
class PromptConfig(BaseModel):
    name: str                           # Unique identifier
    version: str = "1.0"                # Version string
    system: Optional[str] = None        # System message
    role: str = "assistant"             # Role identifier
    template: str                       # Jinja2 template
    variables: List[str] = []           # Required variables
    metadata: Dict[str, Any] = {}       # Additional metadata
    few_shot_examples: Optional[List[FewShotExample]] = None
    generation_config: Optional[GenerationConfig] = None
```

**Example**:
```python
from config.schema import PromptConfig

config = PromptConfig(
    name="qa_assistant",
    template="Question: {{ question }}\nAnswer:",
    variables=["question"]
)
```

---

### `ConfigLoader`

Utility for loading configurations from YAML files.

**Methods**:

#### `load_prompt_config(path: Union[str, Path]) -> PromptConfig`

Load and validate prompt configuration from YAML.

**Parameters**:
- `path`: Path to YAML file

**Returns**: Validated `PromptConfig` object

**Raises**:
- `FileNotFoundError`: If file doesn't exist
- `ValidationError`: If config doesn't match schema

**Example**:
```python
from config.loader import ConfigLoader

config = ConfigLoader.load_prompt_config("configs/prompts/qa.yaml")
```

---

## Prompts Module (`src/prompts/`)

### `PromptTemplate`

Handles prompt rendering with variable substitution.

**Constructor**:
```python
def __init__(self, config: PromptConfig)
```

**Methods**:

#### `render(**kwargs: Any) -> str`

Render prompt with provided variables.

**Parameters**:
- `**kwargs`: Variable key-value pairs

**Returns**: Rendered prompt string

**Raises**:
- `ValueError`: If required variables are missing

**Example**:
```python
from prompts.template import PromptTemplate

template = PromptTemplate(config)
rendered = template.render(question="What is AI?")
```

#### `render_with_system(**kwargs: Any) -> str`

Render prompt with system message prepended.

**Example**:
```python
full_prompt = template.render_with_system(question="What is AI?")
# Returns: "{system message}\n\nQuestion: What is AI?\nAnswer:"
```

#### `render_with_few_shot(include_examples: bool = True, **kwargs: Any) -> str`

Render prompt with few-shot examples.

**Parameters**:
- `include_examples`: Whether to include examples
- `**kwargs`: Variable key-value pairs

**Example**:
```python
few_shot_prompt = template.render_with_few_shot(
    question="What is machine learning?"
)
```

#### `get_required_variables() -> List[str]`

Get list of required template variables.

**Returns**: List of variable names

#### `validate_variables(**kwargs: Any) -> bool`

Check if all required variables are provided.

**Returns**: True if all present, False otherwise

---

## Evaluators Module (`src/evaluators/`)

*Coming soon - will document as implemented*

---

## Models Module (`src/models/`)

*Coming soon - will document as implemented*

---

## Experiments Module (`src/experiments/`)

*Coming soon - will document as implemented*

---

## Usage Patterns

### Basic Workflow

```python
from config.loader import ConfigLoader
from prompts.template import PromptTemplate

# 1. Load configuration
config = ConfigLoader.load_prompt_config("configs/prompts/qa.yaml")

# 2. Create template
template = PromptTemplate(config)

# 3. Render prompt
prompt = template.render(
    question="What is the capital of France?"
)

# 4. Use with model (to be implemented)
# model = ModelLoader.load("llama-2-7b")
# response = model.generate(prompt)
```

### Advanced: Few-Shot Learning

```python
# Load config with few-shot examples
config = ConfigLoader.load_prompt_config("configs/prompts/with_examples.yaml")

template = PromptTemplate(config)

# Render with examples
prompt = template.render_with_few_shot(
    question="Who wrote Romeo and Juliet?",
    include_examples=True
)
```

### Configuration in Code

```python
from config.schema import PromptConfig, FewShotExample

# Create config programmatically
config = PromptConfig(
    name="custom_qa",
    version="1.0",
    system="You are a helpful assistant.",
    template="Q: {{ question }}\nA:",
    variables=["question"],
    few_shot_examples=[
        FewShotExample(
            input="What is 2+2?",
            output="4"
        )
    ]
)

# Save to file
ConfigLoader.save_prompt_config(config, "configs/prompts/custom.yaml")
```

---

## Error Handling

### Common Errors

**Missing Variables**:
```python
try:
    template.render()  # Missing 'question'
except ValueError as e:
    print(f"Error: {e}")
    # Error: Missing required variables: {'question'}
```

**Invalid Template**:
```python
try:
    config = PromptConfig(
        name="bad",
        template="{{ invalid syntax }",
        variables=[]
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

**File Not Found**:
```python
try:
    config = ConfigLoader.load_prompt_config("nonexistent.yaml")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

---

## Type Hints

All functions include type hints for better IDE support:

```python
def render(self, **kwargs: Any) -> str: ...
def load_prompt_config(path: Union[str, Path]) -> PromptConfig: ...
```

Use `mypy` for static type checking:
```bash
mypy src/
```
