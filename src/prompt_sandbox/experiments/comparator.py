"""
Result comparison and statistical analysis
"""

import statistics
from typing import List, Dict, Any, Tuple
from .runner import ExperimentResult


class ResultComparator:
    """
    Compare experiment results with statistical analysis

    Helps identify which prompts/models perform better with significance testing.
    """

    def __init__(self, results: List[ExperimentResult]):
        """
        Initialize comparator with results

        Args:
            results: List of ExperimentResult objects or dicts from storage
        """
        self.results = results

    @staticmethod
    def _get_field(result, field_name: str):
        """Helper to get field from either dataclass or dict"""
        if isinstance(result, dict):
            return result.get(field_name)
        return getattr(result, field_name, None)

    def compare_prompts(
        self,
        model_name: str,
        metric: str = "bleu"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different prompts for the same model

        Args:
            model_name: Model to compare prompts on
            metric: Evaluation metric to compare

        Returns:
            Dict mapping prompt names to statistics
        """
        # Group results by prompt
        prompt_scores: Dict[str, List[float]] = {}

        for result in self.results:
            result_model = self._get_field(result, 'model_name')
            result_prompt = self._get_field(result, 'prompt_name')
            result_scores = self._get_field(result, 'evaluation_scores')

            if result_model == model_name:
                if result_prompt not in prompt_scores:
                    prompt_scores[result_prompt] = []

                if result_scores and metric in result_scores:
                    prompt_scores[result_prompt].append(result_scores[metric])

        # Calculate statistics for each prompt
        comparison = {}
        for prompt, scores in prompt_scores.items():
            if scores:
                comparison[prompt] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }

        return comparison

    def compare_models(
        self,
        prompt_name: str,
        metric: str = "bleu"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different models for the same prompt

        Args:
            prompt_name: Prompt to compare models on
            metric: Evaluation metric to compare

        Returns:
            Dict mapping model names to statistics
        """
        # Group results by model
        model_scores: Dict[str, List[float]] = {}

        for result in self.results:
            result_model = self._get_field(result, 'model_name')
            result_prompt = self._get_field(result, 'prompt_name')
            result_scores = self._get_field(result, 'evaluation_scores')

            if result_prompt == prompt_name:
                if result_model not in model_scores:
                    model_scores[result_model] = []

                if result_scores and metric in result_scores:
                    model_scores[result_model].append(result_scores[metric])

        # Calculate statistics for each model
        comparison = {}
        for model, scores in model_scores.items():
            if scores:
                comparison[model] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }

        return comparison

    def get_best_prompt(
        self,
        model_name: str,
        metric: str = "bleu"
    ) -> Tuple[str, float]:
        """
        Find best performing prompt for a model

        Args:
            model_name: Model to evaluate prompts on
            metric: Metric to optimize for

        Returns:
            Tuple of (prompt_name, mean_score)
        """
        comparison = self.compare_prompts(model_name, metric)

        if not comparison:
            return ("", 0.0)

        best_prompt = max(comparison.items(), key=lambda x: x[1]["mean"])
        return (best_prompt[0], best_prompt[1]["mean"])

    def get_best_model(
        self,
        prompt_name: str,
        metric: str = "bleu"
    ) -> Tuple[str, float]:
        """
        Find best performing model for a prompt

        Args:
            prompt_name: Prompt to evaluate models on
            metric: Metric to optimize for

        Returns:
            Tuple of (model_name, mean_score)
        """
        comparison = self.compare_models(prompt_name, metric)

        if not comparison:
            return ("", 0.0)

        best_model = max(comparison.items(), key=lambda x: x[1]["mean"])
        return (best_model[0], best_model[1]["mean"])

    def generate_report(self) -> str:
        """
        Generate formatted comparison report

        Returns:
            Markdown-formatted report string
        """
        report_lines = ["# Experiment Comparison Report\n"]

        # Get unique prompts and models
        prompts = set(r.prompt_name for r in self.results)
        models = set(r.model_name for r in self.results)

        report_lines.append(f"**Prompts**: {len(prompts)}")
        report_lines.append(f"**Models**: {len(models)}")
        report_lines.append(f"**Total Results**: {len(self.results)}\n")

        # Best combinations
        report_lines.append("## Best Performers\n")

        for model in models:
            best_prompt, score = self.get_best_prompt(model, "bleu")
            if best_prompt:
                report_lines.append(
                    f"- **{model}**: Best prompt = `{best_prompt}` "
                    f"(BLEU: {score:.2f})"
                )

        report_lines.append("")

        for prompt in prompts:
            best_model, score = self.get_best_model(prompt, "bleu")
            if best_model:
                report_lines.append(
                    f"- **{prompt}**: Best model = `{best_model}` "
                    f"(BLEU: {score:.2f})"
                )

        return "\n".join(report_lines)
