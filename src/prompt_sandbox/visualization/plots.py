"""
Visualization tools for experiment results
"""

from typing import List, Optional
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ResultVisualizer:
    """
    Create visualizations from experiment results

    Generates charts comparing prompts, models, and metrics.
    """

    def __init__(self, results_data: List[dict]):
        """
        Initialize visualizer with results

        Args:
            results_data: List of result dictionaries
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        self.results = results_data

    def plot_prompt_comparison(
        self,
        model_name: str,
        metric: str = "bleu",
        output_path: Optional[Path] = None
    ):
        """
        Create bar chart comparing prompts for a model

        Args:
            model_name: Model to compare prompts on
            metric: Metric to visualize
            output_path: Path to save plot (None = show)
        """
        # Group by prompt
        prompt_scores = {}
        for result in self.results:
            if result["model_name"] == model_name:
                prompt = result["prompt_name"]
                if prompt not in prompt_scores:
                    prompt_scores[prompt] = []
                if metric in result["evaluation_scores"]:
                    prompt_scores[prompt].append(result["evaluation_scores"][metric])

        if not prompt_scores:
            print(f"No results found for model: {model_name}")
            return

        # Calculate means
        prompts = list(prompt_scores.keys())
        means = [sum(scores)/len(scores) for scores in prompt_scores.values()]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(prompts, means, color='steelblue', alpha=0.8)
        ax.set_xlabel('Prompt', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Prompt Comparison - {model_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_model_comparison(
        self,
        prompt_name: str,
        metric: str = "bleu",
        output_path: Optional[Path] = None
    ):
        """
        Create bar chart comparing models for a prompt

        Args:
            prompt_name: Prompt to compare models on
            metric: Metric to visualize
            output_path: Path to save plot (None = show)
        """
        # Group by model
        model_scores = {}
        for result in self.results:
            if result["prompt_name"] == prompt_name:
                model = result["model_name"]
                if model not in model_scores:
                    model_scores[model] = []
                if metric in result["evaluation_scores"]:
                    model_scores[model].append(result["evaluation_scores"][metric])

        if not model_scores:
            print(f"No results found for prompt: {prompt_name}")
            return

        # Calculate means
        models = list(model_scores.keys())
        means = [sum(scores)/len(scores) for scores in model_scores.values()]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(models, means, color='forestgreen', alpha=0.8)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {prompt_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_metric_heatmap(
        self,
        metric: str = "bleu",
        output_path: Optional[Path] = None
    ):
        """
        Create heatmap of prompts vs models

        Args:
            metric: Metric to visualize
            output_path: Path to save plot
        """
        import numpy as np

        # Build matrix
        prompts = sorted(set(r["prompt_name"] for r in self.results))
        models = sorted(set(r["model_name"] for r in self.results))

        matrix = np.zeros((len(prompts), len(models)))

        for i, prompt in enumerate(prompts):
            for j, model in enumerate(models):
                scores = [
                    r["evaluation_scores"].get(metric, 0)
                    for r in self.results
                    if r["prompt_name"] == prompt and r["model_name"] == model
                ]
                matrix[i, j] = sum(scores) / len(scores) if scores else 0

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(prompts)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(prompts)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric.upper()} Score', fontweight='bold')

        # Add values
        for i in range(len(prompts)):
            for j in range(len(models)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        ax.set_title(f'Prompt vs Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 Heatmap saved to: {output_path}")
        else:
            plt.show()

        plt.close()
