# Prompt-Sandbox

A production-ready framework for systematic prompt engineering with LLMs. Compare prompts across models, evaluate with multiple metrics, and discover insights through reproducible experiments.

## Features

- **Config-Driven Prompts**: YAML-based prompt templates with Jinja2 rendering
- **Multi-Model Support**: Hugging Face Transformers, vLLM, Ollama (local), OpenAI API (optional)
- **Comprehensive Evaluation**: BLEU, BERTScore, ROUGE, faithfulness checking
- **Async Execution**: Parallel experiment runs with batching
- **Results Analysis**: Side-by-side comparisons with statistical testing

**Note**: This project works 100% locally with free models (Ollama, Hugging Face). OpenAI API is **optional** for users who want to compare against GPT models.

## Installation

```bash
# Clone the repository
git clone https://github.com/vihanga/prompt-sandbox.git
cd prompt-sandbox

# Install in development mode
pip install -e .

# Optional: Install dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from prompt_sandbox.config.loader import ConfigLoader
from prompt_sandbox.prompts.template import PromptTemplate

# Load prompt configuration
config = ConfigLoader.load_prompt_config("configs/prompts/qa_assistant.yaml")

# Create template
template = PromptTemplate(
    template=config.template,
    system=config.system
)

# Render with variables
prompt = template.render(
    question="What is the capital of France?",
    context="France is a country in Western Europe."
)

print(prompt)
```

## Project Structure

```
src/prompt_sandbox/          # Main package
├── config/                  # Configuration management
├── models/                  # LLM backends
├── prompts/                 # Template rendering
├── evaluators/              # Evaluation metrics
├── experiments/             # Experiment orchestration
└── utils/                   # Utilities

configs/                     # YAML configurations
├── prompts/                 # Prompt templates
├── experiments/             # Experiment configs
└── models/                  # Model configs

examples/                    # Example scripts
tests/                       # Test suite
_dev/                        # Development documentation
```

## Usage Examples

### Define a Prompt

Create `configs/prompts/custom.yaml`:

```yaml
name: "custom_prompt_v1"
version: "1.0"

system: |
  You are a helpful AI assistant.

template: |
  Question: {{ question }}
  Answer:

variables:
  - question
```

### Run an Experiment

```python
from prompt_sandbox.experiments.runner import ExperimentRunner, ExperimentConfig
from prompt_sandbox.evaluators import BLEUEvaluator, BERTScoreEvaluator

config = ExperimentConfig(
    prompt_configs=["configs/prompts/qa_assistant.yaml"],
    models=[
        "ollama/llama2",  # Free local model via Ollama
        "meta-llama/Llama-2-7b-hf",  # Hugging Face (free)
        # "openai/gpt-4",  # Optional: requires OpenAI API key
    ],
    eval_dataset="data/truthfulqa.json",
    evaluators=[BLEUEvaluator(), BERTScoreEvaluator()],
    num_samples=100
)

runner = ExperimentRunner(config)
results = await runner.run()
```

**Using OpenAI (optional)**:
- Install: `pip install openai`
- Set: `export OPENAI_API_KEY=your_key`
- Add to models list: `"openai/gpt-4"`

Works fine without it!

## Development

See [_dev/architecture.md](_dev/architecture.md) for detailed technical documentation.

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Roadmap

- [x] Core infrastructure (config, templates, models)
- [ ] Evaluation suite (BLEU, BERTScore, faithfulness)
- [ ] Experiment system (async runner, results storage)
- [ ] Comparison tools (visualization, statistical tests)
- [ ] Web UI (Streamlit dashboard)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
