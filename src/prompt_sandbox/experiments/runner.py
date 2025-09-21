"""
Experiment runner for orchestrating prompt evaluations
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.base import ModelBackend, GenerationResult
from ..prompts.template import PromptTemplate
from ..evaluators.base import Evaluator, EvaluationResult


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""

    name: str
    prompts: List[PromptTemplate]
    models: List[ModelBackend]
    evaluators: List[Evaluator]
    test_cases: List[Dict[str, Any]]
    description: Optional[str] = None
    save_results: bool = True
    output_dir: Path = field(default_factory=lambda: Path("results"))


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""

    experiment_name: str
    prompt_name: str
    model_name: str
    test_case_id: int
    input_data: Dict[str, Any]
    generated_text: str
    reference_text: str
    generation_time: float
    evaluation_scores: Dict[str, float]
    evaluation_details: Dict[str, EvaluationResult]
    timestamp: float


class ExperimentRunner:
    """
    Orchestrates experiment execution across prompts, models, and evaluators

    Runs systematic A/B testing of prompts with multiple models and evaluation metrics.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: List[ExperimentResult] = []

    def run(self) -> List[ExperimentResult]:
        """
        Run complete experiment (synchronous)

        Returns:
            List of ExperimentResult objects
        """
        print(f"\n🚀 Starting experiment: {self.config.name}")
        print(f"📊 {len(self.config.prompts)} prompts × "
              f"{len(self.config.models)} models × "
              f"{len(self.config.test_cases)} test cases = "
              f"{len(self.config.prompts) * len(self.config.models) * len(self.config.test_cases)} runs\n")

        start_time = time.time()
        total_runs = 0

        # Iterate through all combinations
        for prompt in self.config.prompts:
            print(f"\n📝 Testing prompt: {prompt.name}")

            for model in self.config.models:
                print(f"  🤖 Model: {model.model_name}")

                for idx, test_case in enumerate(self.config.test_cases):
                    total_runs += 1

                    # Render prompt with test case variables
                    rendered_prompt = prompt.render(**test_case["input"])

                    # Generate response
                    generation_result = model.generate(rendered_prompt)

                    # Evaluate against reference
                    reference = test_case.get("expected_output", "")
                    evaluation_scores = {}
                    evaluation_details = {}

                    for evaluator in self.config.evaluators:
                        eval_result = evaluator.evaluate(
                            generated=generation_result.text,
                            reference=reference
                        )
                        evaluation_scores[evaluator.name] = eval_result.score
                        evaluation_details[evaluator.name] = eval_result

                    # Store result
                    result = ExperimentResult(
                        experiment_name=self.config.name,
                        prompt_name=prompt.name,
                        model_name=model.model_name,
                        test_case_id=idx,
                        input_data=test_case["input"],
                        generated_text=generation_result.text,
                        reference_text=reference,
                        generation_time=generation_result.generation_time,
                        evaluation_scores=evaluation_scores,
                        evaluation_details=evaluation_details,
                        timestamp=time.time()
                    )
                    self.results.append(result)

                    # Print progress
                    print(f"    ✓ Test case {idx + 1}/{len(self.config.test_cases)} "
                          f"({', '.join(f'{k}={v:.3f}' for k, v in evaluation_scores.items())})")

        total_time = time.time() - start_time

        print(f"\n✅ Experiment complete!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Total runs: {total_runs}")
        print(f"⚡ Avg time per run: {total_time/total_runs:.2f}s")

        # Save results if configured
        if self.config.save_results:
            self._save_results()

        return self.results

    def _save_results(self):
        """Save results to JSON file"""
        import json
        from datetime import datetime

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.json"
        filepath = self.config.output_dir / filename

        # Convert results to dict
        results_data = []
        for result in self.results:
            results_data.append({
                "experiment_name": result.experiment_name,
                "prompt_name": result.prompt_name,
                "model_name": result.model_name,
                "test_case_id": result.test_case_id,
                "input_data": result.input_data,
                "generated_text": result.generated_text,
                "reference_text": result.reference_text,
                "generation_time": result.generation_time,
                "evaluation_scores": result.evaluation_scores,
                "timestamp": result.timestamp
            })

        with open(filepath, 'w') as f:
            json.dump({
                "experiment_name": self.config.name,
                "description": self.config.description,
                "num_results": len(results_data),
                "results": results_data
            }, f, indent=2)

        print(f"\n💾 Results saved to: {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all results

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            return {}

        # Aggregate by prompt and model
        summary = {}

        for result in self.results:
            key = (result.prompt_name, result.model_name)

            if key not in summary:
                summary[key] = {
                    "prompt": result.prompt_name,
                    "model": result.model_name,
                    "num_runs": 0,
                    "avg_generation_time": 0,
                    "scores": {evaluator.name: [] for evaluator in self.config.evaluators}
                }

            summary[key]["num_runs"] += 1
            summary[key]["avg_generation_time"] += result.generation_time

            for metric, score in result.evaluation_scores.items():
                summary[key]["scores"][metric].append(score)

        # Calculate averages
        for key, data in summary.items():
            data["avg_generation_time"] /= data["num_runs"]

            for metric, scores in data["scores"].items():
                avg_score = sum(scores) / len(scores) if scores else 0
                data["scores"][metric] = {
                    "mean": avg_score,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                    "count": len(scores)
                }

        return summary
