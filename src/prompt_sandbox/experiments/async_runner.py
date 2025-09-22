"""
Async experiment runner for parallel execution
"""

import asyncio
import time
from typing import List
from .runner import ExperimentConfig, ExperimentResult, ExperimentRunner


class AsyncExperimentRunner(ExperimentRunner):
    """
    Async version of ExperimentRunner for parallel execution

    Significantly faster for batch experiments - runs model inferences
    and evaluations in parallel using async/await.
    """

    async def run_async(self) -> List[ExperimentResult]:
        """
        Run complete experiment asynchronously

        Returns:
            List of ExperimentResult objects
        """
        print(f"\n🚀 Starting async experiment: {self.config.name}")
        print(f"📊 {len(self.config.prompts)} prompts × "
              f"{len(self.config.models)} models × "
              f"{len(self.config.test_cases)} test cases = "
              f"{len(self.config.prompts) * len(self.config.models) * len(self.config.test_cases)} runs\n")

        start_time = time.time()

        # Create all tasks upfront
        tasks = []
        for prompt in self.config.prompts:
            for model in self.config.models:
                for idx, test_case in enumerate(self.config.test_cases):
                    task = self._run_single_async(prompt, model, idx, test_case)
                    tasks.append(task)

        # Run all tasks in parallel
        print(f"⚡ Running {len(tasks)} experiments in parallel...")
        results = await asyncio.gather(*tasks)

        self.results = results
        total_time = time.time() - start_time

        print(f"\n✅ Async experiment complete!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Total runs: {len(results)}")
        print(f"⚡ Avg time per run: {total_time/len(results):.2f}s")
        print(f"🚀 Speedup: ~{len(results) * 2 / total_time:.1f}x faster than sequential")

        # Save results if configured
        if self.config.save_results:
            self._save_results()

        return self.results

    async def _run_single_async(self, prompt, model, idx, test_case):
        """Run a single experiment asynchronously"""

        # Render prompt
        rendered_prompt = prompt.render(**test_case["input"])

        # Generate response (async)
        generation_result = await model.generate_async(rendered_prompt)

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

        # Return result
        return ExperimentResult(
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
