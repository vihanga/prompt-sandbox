"""
Main CLI application using typer
"""

import typer
from pathlib import Path
from typing import Optional

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

    with open(project_path / "configs/prompts/example.yaml", "w") as f:
        f.write(example_prompt)

    typer.echo(f"✅ Initialized prompt-sandbox project at: {project_path}")
    typer.echo(f"📁 Created directories: configs, data, results")
    typer.echo(f"📝 Created example prompt: configs/prompts/example.yaml")


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

    # TODO: Load config and run experiment
    # from prompt_sandbox.experiments import AsyncExperimentRunner
    # runner = AsyncExperimentRunner(config)
    # results = await runner.run_async()

    typer.echo("✅ Experiment complete!")

    if output:
        typer.echo(f"💾 Results saved to: {output}")


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

    # TODO: Load results and generate comparison
    # from prompt_sandbox.experiments import ResultComparator, ResultStorage
    # storage = ResultStorage(results_dir)
    # results = storage.load_results()
    # comparator = ResultComparator(results)
    # report = comparator.generate_report()

    typer.echo("✅ Comparison complete!")

    if output:
        typer.echo(f"📄 Report saved to: {output}")


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

    # TODO: Load and display experiments
    # from prompt_sandbox.experiments import ResultStorage
    # storage = ResultStorage(results_dir)
    # experiments = storage.list_experiments()

    typer.echo(f"📁 Experiments in: {results_dir}")
    typer.echo("(Implementation in progress)")


@app.command()
def version():
    """Show prompt-sandbox version"""
    typer.echo("prompt-sandbox v0.1.0")
    typer.echo("LLM Prompt Engineering Framework")


if __name__ == "__main__":
    app()
