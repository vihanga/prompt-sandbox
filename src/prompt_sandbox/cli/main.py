"""
Main CLI application using typer
"""

import asyncio
import typer
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime

app = typer.Typer(
    name="prompt-sandbox",
    help="LLM Prompt Engineering Framework - Test prompts, evaluate quality, compare results",
    add_completion=False
)


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    path: Optional[Path] = typer.Option(None, help="Project directory (default: current dir)")
):
    """
    Initialize a new prompt-sandbox project

    Creates directory structure and example configs.
    """
    project_path = path or Path(name)
    project_path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (project_path / "configs/prompts").mkdir(parents=True, exist_ok=True)
    (project_path / "configs/experiments").mkdir(parents=True, exist_ok=True)
    (project_path / "data").mkdir(parents=True, exist_ok=True)
    (project_path / "results").mkdir(parents=True, exist_ok=True)

    # Create example prompt config
    example_prompt = """name: "qa_assistant_v1"
version: "1.0"

system: |
  You are a helpful AI assistant that answers questions accurately and concisely.

template: |
  Question: {{ question }}
  Answer:

variables:
  - question
"""

    # Create example experiment config
    example_experiment = """name: "simple_qa_test"

prompts:
  - name: "direct_qa"
    template: "Q: {{question}}\\nA:"
    variables: ["question"]

models:
  - type: "huggingface"
    name: "gpt2"

evaluators:
  - "bleu"
  - "rouge"

test_cases:
  - input:
      question: "What is 2+2?"
    expected_output: "4"
  - input:
      question: "What is the capital of France?"
    expected_output: "Paris"

output_dir: "results"
"""

    with open(project_path / "configs/prompts/example.yaml", "w") as f:
        f.write(example_prompt)

    with open(project_path / "configs/experiments/example.yaml", "w") as f:
        f.write(example_experiment)

    typer.echo(f"✅ Initialized prompt-sandbox project at: {project_path}")
    typer.echo(f"📁 Created directories: configs, data, results")
    typer.echo(f"📝 Created example prompt: configs/prompts/example.yaml")
    typer.echo(f"📝 Created example experiment: configs/experiments/example.yaml")


@app.command()
def eval(
    config: Path = typer.Argument(..., help="Experiment config YAML file"),
    output: Optional[Path] = typer.Option(None, help="Output directory for results")
):
    """
    Run prompt evaluation experiments

    Executes experiments defined in config file and saves results.
    """
    if not config.exists():
        typer.echo(f"❌ Config file not found: {config}", err=True)
        raise typer.Exit(1)

    typer.echo(f"🚀 Running experiment: {config}")
    typer.echo(f"📊 Loading config...")

    try:
        # Load YAML config
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)

        # Import necessary modules
        from prompt_sandbox.config.schema import PromptConfig
        from prompt_sandbox.prompts.template import PromptTemplate
        from prompt_sandbox.models.huggingface import HuggingFaceBackend
        from prompt_sandbox.evaluators import BLEUEvaluator, ROUGEEvaluator, BERTScoreEvaluator
        from prompt_sandbox.experiments import AsyncExperimentRunner, ExperimentConfig

        # Parse prompts
        prompts = []
        for prompt_data in config_data.get('prompts', []):
            prompt_config = PromptConfig(
                name=prompt_data['name'],
                template=prompt_data['template'],
                variables=prompt_data['variables'],
                system=prompt_data.get('system')
            )
            prompts.append(PromptTemplate(prompt_config))

        # Parse models
        models = []
        for model_data in config_data.get('models', []):
            if model_data['type'] == 'huggingface':
                models.append(HuggingFaceBackend(model_data['name']))
            else:
                typer.echo(f"⚠️  Unknown model type: {model_data['type']}", err=True)

        if not models:
            typer.echo("❌ No valid models configured", err=True)
            raise typer.Exit(1)

        # Parse evaluators
        evaluators = []
        evaluator_map = {
            'bleu': BLEUEvaluator,
            'rouge': ROUGEEvaluator,
            'bertscore': BERTScoreEvaluator
        }
        for eval_name in config_data.get('evaluators', []):
            if eval_name.lower() in evaluator_map:
                evaluators.append(evaluator_map[eval_name.lower()]())
            else:
                typer.echo(f"⚠️  Unknown evaluator: {eval_name}")

        # Get test cases
        test_cases = config_data.get('test_cases', [])

        # Determine output directory
        output_dir = Path(output) if output else Path(config_data.get('output_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment config
        experiment_config = ExperimentConfig(
            name=config_data.get('name', 'cli_experiment'),
            prompts=prompts,
            models=models,
            evaluators=evaluators,
            test_cases=test_cases,
            save_results=True,
            output_dir=output_dir
        )

        typer.echo(f"📋 Config: {len(prompts)} prompts, {len(models)} models, {len(test_cases)} test cases")
        typer.echo(f"⚙️  Running experiments...")

        # Run experiment asynchronously
        runner = AsyncExperimentRunner(experiment_config)
        results = asyncio.run(runner.run_async())

        # Get summary
        summary = runner.get_summary()

        typer.echo(f"\n✅ Experiment complete! Generated {len(results)} results")
        typer.echo(f"💾 Results saved to: {output_dir}")

        typer.echo("\n📊 Summary:")
        for (prompt_name, model_name), stats in summary.items():
            typer.echo(f"\n  {prompt_name} + {model_name}:")
            for metric, values in stats['scores'].items():
                typer.echo(f"    {metric.upper()}: {values['mean']:.3f} (±{values['std']:.3f})")

    except Exception as e:
        typer.echo(f"❌ Error running experiment: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def compare(
    results_dir: Path = typer.Argument(..., help="Results directory"),
    metric: str = typer.Option("bleu", help="Metric to compare"),
    output: Optional[Path] = typer.Option(None, help="Output markdown report file")
):
    """
    Compare experiment results

    Analyzes results and generates comparison report.
    """
    if not results_dir.exists():
        typer.echo(f"❌ Results directory not found: {results_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"📊 Comparing results in: {results_dir}")
    typer.echo(f"📈 Using metric: {metric}")

    try:
        from prompt_sandbox.experiments.storage import ResultStorage
        from prompt_sandbox.experiments.comparator import ResultComparator

        # Load results
        storage = ResultStorage(results_dir)

        # Get all experiment files
        experiment_files = list(results_dir.glob("*.json"))
        if not experiment_files:
            typer.echo(f"❌ No experiment results found in {results_dir}", err=True)
            raise typer.Exit(1)

        # Load the most recent experiment (or could prompt user to choose)
        latest_file = max(experiment_files, key=lambda p: p.stat().st_mtime)
        full_name = latest_file.stem

        # Extract base experiment name (remove timestamp suffix if present)
        # Format: experiment_name_YYYYMMDD_HHMMSS -> experiment_name
        parts = full_name.split('_')
        # Find where timestamp starts (8-digit date pattern)
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                experiment_name = '_'.join(parts[:i])
                break
        else:
            experiment_name = full_name

        typer.echo(f"📂 Loading experiment: {full_name}")
        results = storage.load_results(experiment_name)

        if not results:
            typer.echo(f"❌ No results loaded from {experiment_name}", err=True)
            raise typer.Exit(1)

        # Create comparator
        comparator = ResultComparator(results)

        # Get unique prompts and models
        prompts = sorted(set(r['prompt_name'] for r in results))
        models = sorted(set(r['model_name'] for r in results))

        typer.echo(f"\n📋 Found {len(prompts)} prompts and {len(models)} models")

        # Compare prompts for each model
        report_lines = [f"# Experiment Comparison Report: {experiment_name}\n"]
        report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Metric**: {metric.upper()}\n")
        report_lines.append(f"**Results**: {len(results)} total\n")

        for model in models:
            report_lines.append(f"\n## Model: {model}\n")

            try:
                prompt_comparison = comparator.compare_prompts(model, metric)

                report_lines.append("| Prompt | Mean | Std | Min | Max |")
                report_lines.append("|--------|------|-----|-----|-----|")

                for prompt_name, stats in sorted(prompt_comparison.items(),
                                                 key=lambda x: x[1]['mean'],
                                                 reverse=True):
                    report_lines.append(
                        f"| {prompt_name} | {stats['mean']:.3f} | {stats['stdev']:.3f} | "
                        f"{stats['min']:.3f} | {stats['max']:.3f} |"
                    )

                # Find best prompt
                best_prompt, best_score = comparator.get_best_prompt(model, metric)
                report_lines.append(f"\n**🏆 Best Prompt**: {best_prompt} (score: {best_score:.3f})\n")

            except Exception as e:
                report_lines.append(f"\n⚠️  Could not compare prompts: {e}\n")

        # Compare models for each prompt
        report_lines.append("\n## Model Comparison\n")

        for prompt in prompts:
            report_lines.append(f"\n### Prompt: {prompt}\n")

            try:
                model_comparison = comparator.compare_models(prompt, metric)

                report_lines.append("| Model | Mean | Std | Min | Max |")
                report_lines.append("|-------|------|-----|-----|-----|")

                for model_name, stats in sorted(model_comparison.items(),
                                               key=lambda x: x[1]['mean'],
                                               reverse=True):
                    report_lines.append(
                        f"| {model_name} | {stats['mean']:.3f} | {stats['stdev']:.3f} | "
                        f"{stats['min']:.3f} | {stats['max']:.3f} |"
                    )

                # Find best model
                best_model, best_score = comparator.get_best_model(prompt, metric)
                report_lines.append(f"\n**🏆 Best Model**: {best_model} (score: {best_score:.3f})\n")

            except Exception as e:
                report_lines.append(f"\n⚠️  Could not compare models: {e}\n")

        report_content = "\n".join(report_lines)

        # Display summary
        typer.echo("\n✅ Comparison complete!")
        typer.echo(report_content)

        # Save report if requested
        if output:
            output.write_text(report_content)
            typer.echo(f"\n📄 Report saved to: {output}")

    except Exception as e:
        typer.echo(f"❌ Error comparing results: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def list_experiments(
    results_dir: Path = typer.Argument(Path("results"), help="Results directory")
):
    """
    List all saved experiments

    Shows experiment names, timestamps, and result counts.
    """
    if not results_dir.exists():
        typer.echo(f"❌ Results directory not found: {results_dir}", err=True)
        raise typer.Exit(1)

    try:
        from prompt_sandbox.experiments.storage import ResultStorage
        import json

        # Get all JSON files in results directory
        experiment_files = sorted(results_dir.glob("*.json"),
                                 key=lambda p: p.stat().st_mtime,
                                 reverse=True)

        if not experiment_files:
            typer.echo(f"📁 No experiments found in: {results_dir}")
            typer.echo(f"💡 Run 'prompt-sandbox eval <config>' to create experiments")
            return

        typer.echo(f"📁 Experiments in: {results_dir}\n")
        typer.echo(f"{'Name':<30} {'Modified':<20} {'Results':<10}")
        typer.echo("-" * 60)

        for exp_file in experiment_files:
            # Load to count results
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                    result_count = len(data)
            except:
                result_count = "?"

            # Get modification time
            mtime = datetime.fromtimestamp(exp_file.stat().st_mtime)
            mtime_str = mtime.strftime('%Y-%m-%d %H:%M')

            # Display
            name = exp_file.stem
            typer.echo(f"{name:<30} {mtime_str:<20} {result_count:<10}")

        typer.echo(f"\n✅ Found {len(experiment_files)} experiment(s)")

    except Exception as e:
        typer.echo(f"❌ Error listing experiments: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version():
    """Show prompt-sandbox version"""
    typer.echo("prompt-sandbox v0.1.0")
    typer.echo("LLM Prompt Engineering Framework")


if __name__ == "__main__":
    app()
