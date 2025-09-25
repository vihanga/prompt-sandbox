# Prompt-Sandbox

A production-ready framework for systematic prompt engineering with LLMs. Compare prompts across models, evaluate with multiple metrics, and discover insights through reproducible experiments.

## Features

- **Config-Driven Prompts**: YAML-based prompt templates with Jinja2 rendering and variable substitution
- **Multi-Model Support**: Hugging Face Transformers, Ollama (100% local and free), OpenAI API (optional)
- **Comprehensive Evaluation**: BLEU, ROUGE, BERTScore with semantic similarity scoring
- **Async Execution**: Parallel experiment runs with asyncio for 10-50x speedup
- **Results Analysis**: Statistical comparisons, visualization (charts, heatmaps), and best configuration detection
- **Persistent Storage**: JSON and SQLite backends with queryable experiment history
- **CLI Tools**: Command-line interface for project initialization, experiment execution, and result comparison
- **Error Handling**: Automatic retry logic with exponential backoff for API failures

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

### Simple Prompt Template

```python
from prompt_sandbox.config.schema import PromptConfig
from prompt_sandbox.prompts.template import PromptTemplate

# Create a prompt template
config = PromptConfig(
    name="qa_assistant",
    template="Q: {{question}}\nA:",
    variables=["question"]
)

template = PromptTemplate(config)
prompt = template.render(question="What is the capital of France?")
print(prompt)
# Output: Q: What is the capital of France?\nA:
```

### Complete Experiment

```python
import asyncio
from pathlib import Path
from prompt_sandbox.config.schema import PromptConfig
from prompt_sandbox.prompts.template import PromptTemplate
from prompt_sandbox.models import OllamaBackend
from prompt_sandbox.evaluators import BLEUEvaluator, ROUGEEvaluator
from prompt_sandbox.experiments import AsyncExperimentRunner, ExperimentConfig

async def main():
    # Define prompts to test
    prompt1 = PromptTemplate(PromptConfig(
        name="direct_prompt",
        template="Q: {{question}}\nA:",
        variables=["question"]
    ))

    prompt2 = PromptTemplate(PromptConfig(
        name="cot_prompt",
        template="Q: {{question}}\nLet's think step by step:\nA:",
        variables=["question"]
    ))

    # Setup model (requires Ollama running locally)
    model = OllamaBackend("llama3.1")

    # Setup evaluators
    evaluators = [BLEUEvaluator(), ROUGEEvaluator()]

    # Define test cases
    test_cases = [
        {
            "input": {"question": "What is 2+2?"},
            "expected_output": "4"
        },
        {
            "input": {"question": "What is the capital of France?"},
            "expected_output": "Paris"
        }
    ]

    # Configure and run experiment
    config = ExperimentConfig(
        name="prompt_comparison",
        prompts=[prompt1, prompt2],
        models=[model],
        evaluators=evaluators,
        test_cases=test_cases,
        save_results=True,
        output_dir=Path("results")
    )

    runner = AsyncExperimentRunner(config)
    results = await runner.run_async()

    # Get summary and find best prompt
    summary = runner.get_summary()
    print("\n📊 Summary:")
    for key, stats in summary.items():
        print(f"\n{key[0]} + {key[1]}:")
        print(f"  BLEU: {stats['scores']['bleu']['mean']:.2f}")
        print(f"  ROUGE: {stats['scores']['rouge']['mean']:.2f}")

    # Find winner
    from prompt_sandbox.experiments.comparator import ResultComparator
    comparator = ResultComparator(results)
    best_prompt, score = comparator.get_best_prompt("llama3.1", "bleu")
    print(f"\n🏆 Best prompt: {best_prompt} (BLEU: {score:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

```
src/prompt_sandbox/          # Main package
├── config/                  # Configuration management (schema, validation, loading)
├── models/                  # LLM backends (base, HuggingFace, Ollama)
├── prompts/                 # Template rendering (Jinja2-based)
├── evaluators/              # Evaluation metrics (BLEU, ROUGE, BERTScore)
├── experiments/             # Experiment orchestration
│   ├── runner.py            # Sync experiment runner
│   ├── async_runner.py      # Async experiment runner (10-50x faster)
│   ├── storage.py           # Result persistence (JSON/SQLite)
│   └── comparator.py        # Statistical comparison tools
├── visualization/           # Plotting and visualization (matplotlib)
├── cli/                     # Command-line interface (typer)
└── utils/                   # Utilities (retry logic, etc.)

examples/                    # Example scripts
├── complete_example.py      # Full async experiment workflow
tests/                       # Test suite
├── test_evaluators.py       # Unit tests for metrics
└── test_integration.py      # Integration tests for workflows
_dev/                        # Development documentation and planning
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

### Compare Multiple Prompts

```python
from prompt_sandbox.experiments.comparator import ResultComparator
from prompt_sandbox.experiments.storage import ResultStorage

# Load saved results
storage = ResultStorage(Path("results"))
results = storage.load_results("my_experiment")

# Compare prompts for a specific model
comparator = ResultComparator(results)
best_prompt, score = comparator.get_best_prompt("llama3.1", "bleu")
print(f"Best prompt: {best_prompt} (BLEU: {score:.2f})")

# Compare all prompts statistically
comparison = comparator.compare_prompts("llama3.1", "bleu")
for prompt, stats in comparison.items():
    print(f"{prompt}: {stats['mean']:.3f} ± {stats['std']:.3f}")
```

### Visualize Results

```python
from prompt_sandbox.visualization import ResultVisualizer

# Create visualizer from results
visualizer = ResultVisualizer(results)

# Plot prompt comparison
visualizer.plot_prompt_comparison(
    model_name="llama3.1",
    metric="bleu",
    output_path=Path("prompt_comparison.png")
)

# Create heatmap of prompts vs models
visualizer.plot_metric_heatmap(
    metric="bleu",
    output_path=Path("heatmap.png")
)
```

### CLI Usage

```bash
# Initialize new project
prompt-sandbox init my_project

# Run experiment from config
prompt-sandbox eval configs/experiments/experiment1.yaml

# Compare results
prompt-sandbox compare results/ --metric bleu --output report.md

# List all experiments
prompt-sandbox list-experiments results/

# Show version
prompt-sandbox version
```

## Development

See `_dev/prompt-sandbox/` for detailed planning documentation including:
- `PLAN.md` - Development roadmap with task tracking
- `SCHEDULE.md` - Commit schedule and timeline
- `CONTEXT.md` - Architecture and technical decisions
- `RESEARCH.md` - Analysis of similar tools and patterns

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/prompt_sandbox --cov-report=html tests/

# Run only unit tests
pytest -m unit tests/

# Run only integration tests
pytest -m integration tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Roadmap

### Phase 1: Core Framework (✅ Complete)
- [x] Core infrastructure (config, templates, models)
- [x] Model backends (HuggingFace, Ollama)
- [x] Evaluation suite (BLEU, ROUGE, BERTScore)
- [x] Experiment system (async runner, results storage)
- [x] Comparison tools (visualization, statistical tests)
- [x] CLI framework (typer-based)
- [x] Integration tests and quality checks

### Phase 2: Advanced Features (Planned)
- [ ] Web UI (Streamlit dashboard)
- [ ] Additional evaluators (faithfulness, toxicity, custom metrics)
- [ ] Prompt optimization algorithms (genetic, gradient-based)
- [ ] Multi-turn conversation support
- [ ] RAG integration for context-aware prompts
- [ ] Distributed experiment execution
- [ ] Real-time monitoring dashboard

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
